# -*- coding: utf-8 -*-
"""
Paper2CAD Connectivity Inspector Backend Service (Phase 3).

Provides comprehensive inspection and correction services for CircuitGraph:
- Component & terminal inspection with connected node mappings
- Node / Net topology inspection with connected components
- Neighborhood tracing (selected component -> connected wires -> neighboring components)
- Node neighborhood tracing (selected node -> attached terminals -> component list)
- Uncertainty reports with severity, evidence, and actions
- User correction mutations (value, type, terminal reconnection, node merge/split)
  that update the underlying circuit model directly and trigger revalidation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from circuit_data_model import CircuitGraph, ComponentModel, NodeModel, TerminalModel, UncertaintyRecord


class ConnectivityInspector:
    """
    Inspector service operating over a CircuitGraph instance.
    """

    def __init__(self, graph: CircuitGraph):
        self.graph = graph

    # -------------------------------------------------------------
    # Inspection Queries
    # -------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """Returns high-level graph metrics and breakdown."""
        type_counts: Dict[str, int] = {}
        for c in self.graph.components:
            t = c.display_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "image_name": self.graph.image_name,
            "dimensions": {"width": self.graph.width, "height": self.graph.height},
            "total_components": len(self.graph.components),
            "total_nodes": len(self.graph.nodes),
            "total_wires": len(self.graph.wires),
            "total_junctions": len(self.graph.junctions),
            "total_uncertainties": len(self.graph.uncertainties),
            "type_breakdown": type_counts,
            "has_ground": any(n.is_ground for n in self.graph.nodes)
        }

    def inspect_components(self) -> List[Dict[str, Any]]:
        """
        Lists every detected component, its terminals, and connected nodes.
        """
        res = []
        for c in self.graph.components:
            terms = []
            for t in c.terminals:
                terms.append({
                    "terminal_id": t.id,
                    "name": t.name,
                    "side": t.side,
                    "sketch_coordinates": {"x": t.x, "y": t.y},
                    "connected_node_id": t.connected_node_id,
                    "confidence": t.confidence
                })

            res.append({
                "id": c.id,
                "type": c.type,
                "type_refined": c.type_refined,
                "display_type": c.display_type,
                "subtype": c.subtype,
                "subtype_conf": c.subtype_conf,
                "label": c.label,
                "value": c.value,
                "confidence": c.confidence,
                "bbox": c.bbox,
                "center": list(c.center),
                "terminal_count": len(c.terminals),
                "terminals": terms
            })
        return res

    def inspect_nodes(self) -> List[Dict[str, Any]]:
        """
        Lists each electrical node and all connected component terminals.
        """
        res = []
        for n in self.graph.nodes:
            conns = []
            for conn in n.connections:
                comp = self.graph.get_component(conn.component_id)
                conns.append({
                    "component_id": conn.component_id,
                    "component_type": comp.display_type if comp else "unknown",
                    "component_label": (comp.label or comp.value) if comp else None,
                    "terminal_id": conn.terminal_id,
                    "terminal_name": conn.terminal_name,
                    "confidence": conn.confidence
                })

            res.append({
                "node_id": n.id,
                "name": n.name,
                "is_ground": n.is_ground,
                "connected_component_count": len(conns),
                "connections": conns
            })
        return res

    def inspect_component_neighborhood(self, comp_id: int) -> Optional[Dict[str, Any]]:
        """
        Highlights a selected component, its connected wires, and all adjacent components.
        """
        comp = self.graph.get_component(comp_id)
        if not comp:
            return None

        # Connected node IDs
        connected_node_ids = [t.connected_node_id for t in comp.terminals if t.connected_node_id is not None]

        # Adjacent components connected via these nodes
        adjacent_components = []
        adjacent_comp_ids = set()
        for nid in connected_node_ids:
            for other_c in self.graph.get_connected_components(nid):
                if other_c.id != comp_id and other_c.id not in adjacent_comp_ids:
                    adjacent_comp_ids.add(other_c.id)
                    adjacent_components.append({
                        "id": other_c.id,
                        "type": other_c.display_type,
                        "label": other_c.label or other_c.value,
                        "via_node_id": nid
                    })

        return {
            "component": {
                "id": comp.id,
                "type": comp.display_type,
                "bbox": comp.bbox,
                "terminals": [
                    {"id": t.id, "name": t.name, "x": t.x, "y": t.y, "node_id": t.connected_node_id}
                    for t in comp.terminals
                ]
            },
            "connected_node_ids": connected_node_ids,
            "adjacent_components": adjacent_components
        }

    def inspect_node_neighborhood(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Highlights every component and terminal connected to a selected node.
        """
        node = self.graph.get_node(node_id)
        if not node:
            return None

        connected_comps = self.graph.get_connected_components(node_id)
        return {
            "node_id": node.id,
            "node_name": node.name,
            "is_ground": node.is_ground,
            "components": [
                {
                    "id": c.id,
                    "type": c.display_type,
                    "bbox": c.bbox,
                    "value": c.value,
                    "terminals": [
                        {"id": t.id, "name": t.name, "x": t.x, "y": t.y}
                        for t in c.terminals if t.connected_node_id == node_id
                    ]
                }
                for c in connected_comps
            ]
        }

    def inspect_uncertainties(self) -> List[Dict[str, Any]]:
        """
        Returns all uncertainty records requiring user review.
        """
        return [u.model_dump() for u in self.graph.uncertainties]

    # -------------------------------------------------------------
    # User Corrections & Revalidation
    # -------------------------------------------------------------
    def apply_correction(self, action: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Applies a user-directed correction directly to the circuit graph representation.
        Actions:
        - 'update_value': {'comp_id': int, 'value': str}
        - 'update_type': {'comp_id': int, 'type': str}
        - 'reconnect_terminal': {'comp_id': int, 'terminal_id': int, 'new_node_id': int}
        - 'set_ground_node': {'node_id': int}
        """
        if action == "update_value":
            cid = params.get("comp_id")
            value = params.get("value")
            if cid is None or value is None:
                return False, "Missing comp_id or value"
            val = str(value).strip()
            ok = self.graph.update_component_value(int(cid), val)
            return ok, f"Updated value of component #{cid} to '{val}'" if ok else f"Component #{cid} not found"

        elif action == "update_type":
            cid = params.get("comp_id")
            new_type = params.get("type", "").strip()
            if cid is None or not new_type:
                return False, "Missing comp_id or type"
            ok = self.graph.update_component_type(int(cid), new_type)
            return ok, f"Updated type of component #{cid} to '{new_type}'" if ok else f"Component #{cid} not found"

        elif action == "reconnect_terminal":
            cid = params.get("comp_id")
            tid = params.get("terminal_id")
            nid = params.get("new_node_id")
            if cid is None or tid is None or nid is None:
                return False, "Missing comp_id, terminal_id, or new_node_id"
            ok = self.graph.update_terminal_connection(int(cid), int(tid), int(nid))
            return ok, f"Reconnected terminal #{tid} of comp #{cid} to node #{nid}" if ok else "Failed to reconnect terminal"

        elif action == "set_ground_node":
            nid = params.get("node_id")
            if nid is None:
                return False, "Missing node_id"
            target_nid = int(nid)
            found = False
            for n in self.graph.nodes:
                if n.id == target_nid:
                    n.is_ground = True
                    n.name = "0"
                    found = True
                else:
                    if n.is_ground and n.id != target_nid:
                        n.is_ground = False
                        n.name = f"N{n.id}"
            return found, f"Node #{target_nid} designated as ground reference (0)" if found else f"Node #{target_nid} not found"

        return False, f"Unknown action: '{action}'"
