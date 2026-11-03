# -*- coding: utf-8 -*-
"""
Paper2CAD Reliable Circuit Data Model (Phase 2).

Provides strongly-typed, normalized internal circuit representation for:
- Components (bounding boxes, recognized types, refined subtypes, values, confidences, terminals)
- Nodes / Nets (electrical equivalence classes, connected component terminals)
- Wires & Junctions (endpoints, pixel coordinates, crossing ambiguities)
- Terminal-to-Node Connections (with confidence metrics)
- Explicit Uncertainty Records (ambiguous types, uncertain values, questionable connections)

Preserves original sketch pixel coordinates for interactive frontend highlighting
and synchronizes with existing repository formats (.final.json, .link.json, SPICE netlists).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field


class TerminalModel(BaseModel):
    """
    Component terminal / pin model.
    """
    id: int
    name: str = "T1"  # "T1", "T2", "+", "-", "G", "D", "S", "IN+", "IN-", "OUT"
    side: Optional[str] = None  # "left", "right", "top", "bottom"
    x: float  # Absolute pixel X in sketch
    y: float  # Absolute pixel Y in sketch
    component_id: int
    connected_node_id: Optional[int] = None
    confidence: float = 1.0


class ComponentModel(BaseModel):
    """
    Physical circuit component detected in the hand-drawn sketch.
    """
    id: int
    cls_id: Optional[int] = None
    type: str  # Base class (e.g. "resistor", "capacitor", "voltage.dc", "gnd")
    type_refined: Optional[str] = None  # Refined class (e.g. "current.dc", "diode.light_emitting")
    subtype: Optional[str] = None  # Fine-grained subtype (e.g. "idc", "npn", "nmos")
    subtype_conf: Optional[float] = None
    label: Optional[str] = None  # Circuit reference designator (e.g. "R1", "C2", "D1")
    value: Optional[str] = None  # Electrical parameter value (e.g. "10kΩ", "100mH", "5V")
    confidence: float = 1.0
    bbox: List[float] = Field(default_factory=list)  # [x1, y1, x2, y2]
    terminals: List[TerminalModel] = Field(default_factory=list)
    crop_url: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        if len(self.bbox) == 4:
            return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)
        return (0.0, 0.0)

    @property
    def display_type(self) -> str:
        return self.type_refined or self.type


class NodeConnection(BaseModel):
    """
    Electrical attachment of a component terminal to a node.
    """
    component_id: int
    terminal_id: int
    terminal_name: str
    confidence: float = 1.0


class NodeModel(BaseModel):
    """
    Electrical node / net connecting multiple component terminals together.
    """
    id: int
    name: str  # "0" for ground, "N1", "N2", etc.
    is_ground: bool = False
    connections: List[NodeConnection] = Field(default_factory=list)
    associated_wire_ids: List[int] = Field(default_factory=list)


class JunctionModel(BaseModel):
    """
    Physical intersection or crossover of wires detected in the sketch.
    """
    id: int
    x: float
    y: float
    type: str = "junction"  # "junction", "crossover", "terminal"
    confidence: float = 1.0
    degree: int = 3
    node_id: Optional[int] = None


class WireSegmentModel(BaseModel):
    """
    Linear conductive wire segment connecting terminals or junctions.
    """
    id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    node_id: Optional[int] = None
    length: float = 0.0
    confidence: float = 1.0
    ambiguous_crossing: bool = False


class UncertaintyRecord(BaseModel):
    """
    Identified ambiguity or low-confidence inference requiring inspection.
    """
    id: str
    target_type: str  # "component", "terminal", "connection", "value", "junction", "topology"
    target_id: int
    severity: str = "warning"  # "info", "warning", "error"
    issue_type: str  # "ambiguous_type", "missing_value", "uncertain_connection", "floating_terminal", "suspicious_crossing"
    confidence: float = 0.5
    description: str
    evidence: str
    suggested_action: str


class CircuitGraph(BaseModel):
    """
    Complete normalized circuit graph model for Paper2CAD.
    Source of truth for inspector, rule validator, schematic renderer, and LLM assistant.
    """
    image_name: str
    width: int
    height: int
    components: List[ComponentModel] = Field(default_factory=list)
    nodes: List[NodeModel] = Field(default_factory=list)
    wires: List[WireSegmentModel] = Field(default_factory=list)
    junctions: List[JunctionModel] = Field(default_factory=list)
    uncertainties: List[UncertaintyRecord] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------
    # Query & Accessor Methods
    # -------------------------------------------------------------
    def get_component(self, comp_id: int) -> Optional[ComponentModel]:
        for c in self.components:
            if c.id == comp_id:
                return c
        return None

    def get_node(self, node_id: int) -> Optional[NodeModel]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_connected_components(self, node_id: int) -> List[ComponentModel]:
        node = self.get_node(node_id)
        if not node:
            return []
        c_ids = {conn.component_id for conn in node.connections}
        return [c for c in self.components if c.id in c_ids]

    def get_component_nodes(self, comp_id: int) -> Dict[str, Optional[int]]:
        comp = self.get_component(comp_id)
        if not comp:
            return {}
        return {term.name: term.connected_node_id for term in comp.terminals}

    # -------------------------------------------------------------
    # Mutation Methods (Preserve Integrity)
    # -------------------------------------------------------------
    def update_component_value(self, comp_id: int, new_value: str) -> bool:
        comp = self.get_component(comp_id)
        if not comp:
            return False
        comp.value = new_value
        # Remove resolved value uncertainty if exists
        self.uncertainties = [
            u for u in self.uncertainties
            if not (u.target_type == "component" and u.target_id == comp_id and u.issue_type == "missing_value")
        ]
        return True

    def update_component_type(self, comp_id: int, new_type: str) -> bool:
        comp = self.get_component(comp_id)
        if not comp:
            return False
        comp.type_refined = new_type
        # Clean ambiguous type uncertainty
        self.uncertainties = [
            u for u in self.uncertainties
            if not (u.target_type == "component" and u.target_id == comp_id and u.issue_type == "ambiguous_type")
        ]
        return True

    def update_terminal_connection(self, comp_id: int, terminal_id: int, new_node_id: int) -> bool:
        comp = self.get_component(comp_id)
        if not comp:
            return False
        term = next((t for t in comp.terminals if t.id == terminal_id), None)
        if not term:
            return False

        old_node_id = term.connected_node_id
        term.connected_node_id = new_node_id

        # Remove from old node
        if old_node_id is not None:
            old_node = self.get_node(old_node_id)
            if old_node:
                old_node.connections = [
                    c for c in old_node.connections
                    if not (c.component_id == comp_id and c.terminal_id == terminal_id)
                ]

        # Add to new node
        new_node = self.get_node(new_node_id)
        if not new_node:
            new_node = NodeModel(
                id=new_node_id,
                name="0" if new_node_id == 0 else f"N{new_node_id}",
                is_ground=(new_node_id == 0),
                connections=[]
            )
            self.nodes.append(new_node)

        new_node.connections.append(NodeConnection(
            component_id=comp_id,
            terminal_id=terminal_id,
            terminal_name=term.name,
            confidence=1.0
        ))
        return True

    # -------------------------------------------------------------
    # Ingestion: from_final_json (Base Model Integration)
    # -------------------------------------------------------------
    @classmethod
    def from_final_json(cls, data_or_path: Union[str, Path, dict]) -> CircuitGraph:
        """
        Parses repository's Stage 8/9 .final.json into strongly-typed CircuitGraph.
        Preserves geometric coordinates, OCR labels, refined types, and flags uncertainties.
        """
        if isinstance(data_or_path, (str, Path)):
            with open(data_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = data_or_path

        # If already formatted as a full CircuitGraph
        if "nodes" in data and "components" in data and not data.get("nets") and not data.get("points"):
            try:
                return cls(**data)
            except Exception:
                pass

        image_name = data.get("image", "circuit")
        width = int(data.get("width", 1000))
        height = int(data.get("height", 800))

        raw_components = data.get("components", [])
        raw_points = data.get("points", [])
        raw_endpoints = data.get("endpoints", [])
        raw_edges = data.get("edges", [])
        raw_texts = data.get("texts", [])
        raw_nets = data.get("nets", [])
        endpoint_to_node_id = {
            eid: int(net.get("id", net_idx))
            for net_idx, net in enumerate(raw_nets)
            for eid in net.get("endpoint_eids", [])
        }

        # Build point coordinate lookup
        point_by_id: Dict[int, dict] = {}
        for p in raw_points:
            point_by_id[p["id"]] = p

        # Map texts to component instances
        comp_text_map: Dict[int, str] = {}
        comp_val_map: Dict[int, str] = {}
        for t in raw_texts:
            matched_id = t.get("matched_component_id")
            token = (t.get("text") or "").strip()
            if matched_id is not None and token:
                # Distinguish reference designator (R1, C1) vs value (10k, 1uF)
                if re.match(r"^[RCLVDQI]\d+$", token, re.IGNORECASE):
                    comp_text_map[int(matched_id)] = token
                else:
                    comp_val_map[int(matched_id)] = token

        # 1. Parse Components and Terminals
        components: List[ComponentModel] = []
        terminal_global_id = 0
        endpoint_to_term_map: Dict[int, Tuple[int, int]] = {}  # eid -> (comp_id, term_id)
        point_id_to_term_map: Dict[int, Tuple[int, int]] = {}  # point_id -> (comp_id, term_id)

        # Build lookup from raw_endpoints: eid -> endpoint
        endpoint_by_eid: Dict[int, dict] = {ep.get("eid", ep.get("id", i)): ep for i, ep in enumerate(raw_endpoints)}

        for rc in raw_components:
            cid = int(rc["id"])
            cls_name = rc.get("cls_name", "unknown")
            cls_refined = rc.get("cls_name_refined")
            subtype = rc.get("subtype")
            subtype_conf = rc.get("subtype_conf")
            bbox = [float(v) for v in rc.get("bbox", [0, 0, 0, 0])]

            comp_terminals: List[TerminalModel] = []

            # A) Check raw_endpoints with comp_id directly
            for ep in raw_endpoints:
                eid = ep.get("eid", ep.get("id"))
                if ep.get("comp_id") == cid or (ep.get("kind") == "terminal" and ep.get("comp_id") == cid):
                    tid = terminal_global_id
                    terminal_global_id += 1
                    side = ep.get("side")
                    t_name = f"T{len(comp_terminals)+1}"
                    if cls_name == "gnd":
                        t_name = "GND"
                    elif "diode" in cls_name or "led" in (cls_refined or ""):
                        t_name = "A" if side in ("left", "top") else "K"
                    elif "transistor" in cls_name or "bjt" in (cls_refined or ""):
                        t_name = "B" if side == "left" else ("C" if side == "top" else "E")

                    term_model = TerminalModel(
                        id=tid,
                        name=t_name,
                        side=side,
                        x=float(ep.get("x", ep.get("x_raw", 0))),
                        y=float(ep.get("y", ep.get("y_raw", 0))),
                        component_id=cid,
                        confidence=float(ep.get("edge_dist", 1.0))
                    )
                    comp_terminals.append(term_model)
                    if eid is not None:
                        endpoint_to_term_map[eid] = (cid, tid)
                    if "point_id" in ep:
                        point_id_to_term_map[ep["point_id"]] = (cid, tid)

            # B) If no terminals found via endpoints, check raw_points matches
            if not comp_terminals:
                for p in raw_points:
                    for m in p.get("matches", []):
                        if int(m.get("comp_id", -1)) == cid:
                            tid = terminal_global_id
                            terminal_global_id += 1
                            side = m.get("side")
                            t_name = f"T{len(comp_terminals)+1}"
                            term_model = TerminalModel(
                                id=tid,
                                name=t_name,
                                side=side,
                                x=float(p.get("x", 0)),
                                y=float(p.get("y", 0)),
                                component_id=cid,
                                confidence=float(m.get("edge_dist", 1.0))
                            )
                            comp_terminals.append(term_model)
                            point_id_to_term_map[p["id"]] = (cid, tid)

            label = comp_text_map.get(cid)
            val = comp_val_map.get(cid)

            comp = ComponentModel(
                id=cid,
                cls_id=rc.get("cls_id"),
                type=cls_name,
                type_refined=cls_refined,
                subtype=subtype,
                subtype_conf=subtype_conf,
                label=label,
                value=val,
                confidence=float(rc.get("confidence", 0.95)),
                bbox=bbox,
                terminals=comp_terminals,
                properties=rc
            )
            components.append(comp)

        # 2. Build Nodes / Nets
        nodes: List[NodeModel] = []
        is_gnd_net_id: Optional[int] = None

        # Check if ground component exists and attach to net 0
        gnd_comps = [c for c in components if c.type == "gnd"]
        gnd_comp_ids = {c.id for c in gnd_comps}

        for net_idx, rnet in enumerate(raw_nets):
            nid = int(rnet.get("id", net_idx))
            ep_eids = rnet.get("endpoint_eids", [])
            c_ids = rnet.get("component_ids", [])

            # Check if this net touches ground
            has_ground = any(cid in gnd_comp_ids for cid in c_ids)
            if has_ground and is_gnd_net_id is None:
                is_gnd_net_id = nid

            node_connections: List[NodeConnection] = []
            for eid in ep_eids:
                if eid in endpoint_to_term_map:
                    cid, tid = endpoint_to_term_map[eid]
                    c = next((comp for comp in components if comp.id == cid), None)
                    t = next((term for term in c.terminals if term.id == tid), None) if c else None
                    if t:
                        t.connected_node_id = nid
                        node_connections.append(NodeConnection(
                            component_id=cid,
                            terminal_id=tid,
                            terminal_name=t.name,
                            confidence=t.confidence
                        ))

            node_name = "0" if (has_ground or nid == is_gnd_net_id) else f"N{nid}"
            nodes.append(NodeModel(
                id=nid,
                name=node_name,
                is_ground=has_ground,
                connections=node_connections
            ))

        # 3. Parse Junctions & Wire Segments
        junctions: List[JunctionModel] = []
        for p in raw_points:
            pt_type = p.get("type", "unknown")
            if pt_type in ("junction", "crossover"):
                junctions.append(JunctionModel(
                    id=int(p["id"]),
                    x=float(p["x"]),
                    y=float(p["y"]),
                    type=pt_type,
                    confidence=float(p.get("score", 0.9)),
                    degree=int(p.get("degree", 3))
                ))

        wires: List[WireSegmentModel] = []
        for edge_idx, re_edge in enumerate(raw_edges):
            eid1 = re_edge.get("eid1")
            eid2 = re_edge.get("eid2")
            endpoint1 = endpoint_by_eid.get(eid1, {})
            endpoint2 = endpoint_by_eid.get(eid2, {})
            pt1 = point_by_id.get(endpoint1.get("point_id", eid1), {})
            pt2 = point_by_id.get(endpoint2.get("point_id", eid2), {})
            x1 = float(endpoint1.get("x", endpoint1.get("x_raw", pt1.get("x", 0))))
            y1 = float(endpoint1.get("y", endpoint1.get("y_raw", pt1.get("y", 0))))
            x2 = float(endpoint2.get("x", endpoint2.get("x_raw", pt2.get("x", 0))))
            y2 = float(endpoint2.get("y", endpoint2.get("y_raw", pt2.get("y", 0))))
            dist = float(re_edge.get("dist", math.hypot(x2 - x1, y2 - y1)))
            node_id1 = endpoint_to_node_id.get(eid1)
            node_id2 = endpoint_to_node_id.get(eid2)

            wires.append(WireSegmentModel(
                id=edge_idx,
                start_x=x1,
                start_y=y1,
                end_x=x2,
                end_y=y2,
                node_id=node_id1 if node_id1 == node_id2 else None,
                length=dist,
                confidence=float(re_edge.get("ratio", 1.0)),
                ambiguous_crossing=(re_edge.get("dir") == "crossover")
            ))

        # 4. Formulate Explicit Uncertainty Records
        uncertainties: List[UncertaintyRecord] = []
        uncertainty_counter = 1

        for c in components:
            # Check for ambiguous subtype
            if c.subtype and c.subtype_conf and c.subtype_conf < 0.75:
                uncertainties.append(UncertaintyRecord(
                    id=f"UNC-{uncertainty_counter}",
                    target_type="component",
                    target_id=c.id,
                    severity="warning",
                    issue_type="ambiguous_type",
                    confidence=float(c.subtype_conf),
                    description=f"Component #{c.id} ({c.type}) subtype '{c.subtype}' has low classification confidence ({c.subtype_conf:.2f}).",
                    evidence=f"DINOv2 confidence: {c.subtype_conf:.3f}",
                    suggested_action=f"Verify if component #{c.id} is {c.subtype} or alternative."
                ))
                uncertainty_counter += 1

            # Check for missing passive values
            if c.type in ("resistor", "capacitor", "inductor") and not c.value:
                uncertainties.append(UncertaintyRecord(
                    id=f"UNC-{uncertainty_counter}",
                    target_type="value",
                    target_id=c.id,
                    severity="info",
                    issue_type="missing_value",
                    confidence=0.5,
                    description=f"Component #{c.id} ({c.type}) does not have a recognized OCR value.",
                    evidence="No adjacent alphanumeric text token detected in sketch.",
                    suggested_action=f"Assign nominal value (e.g., 1kΩ for resistor, 1uF for capacitor) before SPICE simulation."
                ))
                uncertainty_counter += 1

            # Check for floating/missing terminals
            expected_terms = 1 if c.type in ("gnd", "vss") else (3 if "transistor" in c.type else 2)
            if len(c.terminals) < expected_terms:
                uncertainties.append(UncertaintyRecord(
                    id=f"UNC-{uncertainty_counter}",
                    target_type="terminal",
                    target_id=c.id,
                    severity="warning",
                    issue_type="floating_terminal",
                    confidence=0.4,
                    description=f"Component #{c.id} ({c.type}) has {len(c.terminals)} detected terminal(s), expected {expected_terms}.",
                    evidence=f"Geometric port finder found {len(c.terminals)} endpoints.",
                    suggested_action="Check if a wire trace was broken or disconnected in the sketch."
                ))
                uncertainty_counter += 1

        # Check for ambiguous crossovers in wires
        for w in wires:
            if w.ambiguous_crossing:
                uncertainties.append(UncertaintyRecord(
                    id=f"UNC-{uncertainty_counter}",
                    target_type="junction",
                    target_id=w.id,
                    severity="info",
                    issue_type="suspicious_crossing",
                    confidence=0.6,
                    description=f"Wire segment #{w.id} crosses another wire trace without an explicit solder dot.",
                    evidence="HAWP detected non-planar wire crossover.",
                    suggested_action="Confirm whether wires are insulated or intended to be electrically joined."
                ))
                uncertainty_counter += 1

        return cls(
            image_name=image_name,
            width=width,
            height=height,
            components=components,
            nodes=nodes,
            wires=wires,
            junctions=junctions,
            uncertainties=uncertainties,
            metadata={"source": "final_json", "num_points": len(raw_points)}
        )

    # -------------------------------------------------------------
    # Serialization Methods
    # -------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Returns standard serializable dictionary."""
        return self.model_dump()

    def to_final_json(self) -> Dict[str, Any]:
        """Exports back to repository's Stage 8/9 .final.json compatible format."""
        out_comps = []
        for c in self.components:
            cdict = {
                "id": c.id,
                "cls_id": c.cls_id,
                "cls_name": c.type,
                "bbox": c.bbox
            }
            if c.type_refined:
                cdict["cls_name_refined"] = c.type_refined
            if c.subtype:
                cdict["subtype"] = c.subtype
            if c.subtype_conf:
                cdict["subtype_conf"] = c.subtype_conf
            out_comps.append(cdict)

        out_nets = []
        for n in self.nodes:
            out_nets.append({
                "id": n.id,
                "endpoint_eids": [conn.terminal_id for conn in n.connections],
                "component_ids": list({conn.component_id for conn in n.connections})
            })

        out_texts = []
        for c in self.components:
            if c.value or c.label:
                out_texts.append({
                    "image": self.image_name,
                    "text": c.value or c.label,
                    "matched_component_id": c.id,
                    "component_cls_name": c.type
                })

        return {
            "image": self.image_name,
            "width": self.width,
            "height": self.height,
            "components": out_comps,
            "nets": out_nets,
            "texts": out_texts
        }
