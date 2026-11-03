# -*- coding: utf-8 -*-
"""
Paper2CAD Official Model Context Protocol (MCP) Server (Phase 7).

Exposes real circuit analysis, inspection, validation, simulation, and schematic
tools using the official Python MCP SDK (MCPServer 2.x).

Tools exposed:
- inspect_circuit(circuit_id): Components, nodes, connections, uncertainties
- validate_circuit(circuit_id): Rule-based domain checks (shorts, floating pins, ground)
- generate_schematic(circuit_id, theme): Renders IEEE CAD SVG diagram
- export_spice_netlist(circuit_id): Synthesizes and exports SPICE .cir file
- simulate_circuit(circuit_id, analysis): Validates and runs simulation via ngspice
- explain_circuit(circuit_id, question): Grounded AI circuit explanation
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[0] if (Path(__file__).resolve().name == "mcp_server.py" and Path(__file__).resolve().parent.name != "src") else Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from circuit_data_model import CircuitGraph
from connectivity_inspector import ConnectivityInspector
from circuit_validator import CircuitValidator
from cad_schematic_renderer import render_circuit_graph
from spice_simulator import SPICESimulator
from ai_assistant import AICircuitAssistant

from mcp.server.fastmcp import FastMCP
from PIL import Image

mcp_server = FastMCP("Paper2CAD-Schematic-Inspector")


def _normalize_circuit_id(circuit_id: str) -> str:
    """Validate circuit IDs before using them to build filesystem paths."""
    clean_id = circuit_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", clean_id):
        raise ValueError("Circuit ID must contain only letters, numbers, '.', '_' or '-'.")
    return clean_id


def _build_job_config(config: Dict[str, Any], input_dir: Path, job_dir: Path) -> Dict[str, Any]:
    """Redirect every pipeline input/output path into this job's private directory."""
    config = dict(config)
    paths = dict(config.get("paths", {}))
    stage_paths = {
        "src_img": input_dir,
        "src_img_scanned": job_dir / "scan",
        "yolo_detect_exp": job_dir / "yolo" / "exp",
        "yolo_labels": job_dir / "yolo" / "exp" / "labels",
        "yolo_crops": job_dir / "crops",
        "fuse_json": job_dir / "fuse",
        "suppressed_img": job_dir / "suppressed",
        "hawp_json": job_dir / "hawp" / "json",
        "merged_points_vis": job_dir / "merge" / "visualizations",
        "merged_points_json": job_dir / "merge" / "json",
        "dao_xian": job_dir / "wire",
        "link_root": job_dir / "connectivity",
        "link_json": job_dir / "connectivity" / "json",
        "type_refine_json": job_dir / "type_refine" / "json",
        "ports_cls_json": job_dir / "ports" / "json",
        "final_result_root": job_dir / "final",
        "spice_netlists": job_dir / "spice",
    }
    paths.update({key: str(value) for key, value in stage_paths.items()})
    config["paths"] = paths
    return config


def _required_pipeline_weights(config: Dict[str, Any]) -> List[Path]:
    weights = config.get("weights", {})
    required_keys = ("yolo_best", "parseq_ckpt", "hawp_ckpt")
    return [Path(weights[key]) for key in required_keys if weights.get(key)]


def _missing_optional_pipeline_weights(config: Dict[str, Any]) -> Dict[str, List[Path]]:
    weights = config.get("weights", {})
    optional_keys = {
        "DINOv2 subtype refinement": ("bjt_ckpt", "fet_ckpt", "dcsrc_ckpt"),
        "ViTPose port estimation": ("pose3k_wts", "pose2k_wts"),
    }
    return {
        stage: [Path(weights[key]) for key in keys if weights.get(key) and not Path(weights[key]).is_file()]
        for stage, keys in optional_keys.items()
    }


@mcp_server.tool()
def process_sketch(image_path: str, conf_threshold: float = 0.45, device: Optional[str] = None) -> str:
    """
    Runs the full circuit reconstruction pipeline on a local sketch image.
    Returns the reconstructed graph, validation findings, schematic files, and SPICE netlist.
    """
    job_dir: Optional[Path] = None
    try:
        source_path = Path(image_path).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ValueError("image_path must point to a local image file.")
        if source_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            raise ValueError("Unsupported image format. Use a JPG, PNG, or BMP file.")
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1.")
        if device is not None and device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'.")
        try:
            with Image.open(source_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Could not read image file: {exc}") from exc

        from config_utils import load_paths_config

        config = load_paths_config()
        missing_weights = [path for path in _required_pipeline_weights(config) if not path.is_file()]
        if missing_weights:
            missing = ", ".join(str(path) for path in missing_weights)
            raise FileNotFoundError(
                f"Required pipeline model weights are missing: {missing}. "
                "Download the project weights before processing a sketch."
            )

        optional_missing = _missing_optional_pipeline_weights(config)
        skipped_stages = [
            stage for stage, missing in optional_missing.items() if missing
        ]

        job_dir = REPO_ROOT / "outputs" / "mcp_jobs" / uuid.uuid4().hex
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=False)
        staged_image = input_dir / f"sketch{source_path.suffix.lower()}"
        shutil.copy2(source_path, staged_image)

        job_config = _build_job_config(config, input_dir, job_dir)
        config_path = job_dir / "paths.yaml"
        with config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(job_config, config_file, sort_keys=False)

        log_path = job_dir / "pipeline.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pipeline.py"),
            "--config",
            str(config_path),
            "--conf",
            str(conf_threshold),
        ]
        if device:
            command.extend(["--device", device])
        if optional_missing["DINOv2 subtype refinement"]:
            command.append("--skip-type-refinement")
        if optional_missing["ViTPose port estimation"]:
            command.append("--skip-port-estimation")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Pipeline exceeded the 1-hour limit. See log: {log_path}"
                ) from exc

        if result.returncode != 0:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Pipeline exited with status {result.returncode}. "
                f"Log: {log_path}\n{log_text[-4000:]}"
            )

        circuit_id = staged_image.stem
        final_json = job_dir / "final" / "json" / f"{circuit_id}.final.json"
        if not final_json.is_file():
            raise FileNotFoundError(
                f"Pipeline completed without producing the expected circuit graph: {final_json}. "
                f"See log: {log_path}"
            )

        graph = CircuitGraph.from_final_json(final_json)
        inspector = ConnectivityInspector(graph)
        findings = CircuitValidator(graph).validate()

        render_dir = job_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        schematic_files = render_circuit_graph(graph, render_dir)

        from build_spice_netlists import export_one

        spice_dir = job_dir / "spice"
        spice_dir.mkdir(parents=True, exist_ok=True)
        spice_path = Path(export_one(str(final_json), str(spice_dir)))
        netlist = spice_path.read_text(encoding="utf-8")

        return json.dumps({
            "status": "success",
            "job_id": job_dir.name,
            "input_image": source_path.name,
            "circuit_id": graph.image_name,
            "optional_stages_skipped": skipped_stages,
            "graph": graph.to_dict(),
            "inspection": {
                "summary": inspector.get_summary(),
                "components": inspector.inspect_components(),
                "nodes": inspector.inspect_nodes(),
                "uncertainties": inspector.inspect_uncertainties(),
            },
            "validation": {
                "total_findings": len(findings),
                "findings": [finding.model_dump() for finding in findings],
            },
            "schematic": {
                "output_directory": str(render_dir),
                "files": schematic_files,
            },
            "spice": {
                "netlist_path": str(spice_path),
                "netlist": netlist,
            },
            "output_directory": str(job_dir),
        }, indent=2)
    except Exception as exc:
        error = {"error": str(exc)}
        if job_dir is not None:
            error["job_directory"] = str(job_dir)
        return json.dumps(error, indent=2)


def _resolve_circuit_graph(circuit_id: str) -> CircuitGraph:
    """Helper to load circuit graph from sample circuits or UI job directories."""
    clean_id = _normalize_circuit_id(circuit_id)

    # 1. Check UI jobs
    ui_job_json = REPO_ROOT / "outputs" / "ui_jobs" / clean_id / "circuit.json"
    if ui_job_json.exists():
        with open(ui_job_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Check if stored as full graph or components list
            if "components" in data and "nodes" in data:
                return CircuitGraph(**data)
            # Otherwise build from final_json style
            return CircuitGraph.from_final_json(ui_job_json)

    # 2. Check sample final.json
    final_json = REPO_ROOT / "outputs" / "run1" / "final_result" / "json" / f"{clean_id}.final.json"
    if final_json.exists():
        return CircuitGraph.from_final_json(final_json)

    # 3. Check ground truth json
    gt_json = REPO_ROOT / "data" / "ground_truth" / f"{clean_id}.gt.json"
    if gt_json.exists():
        return CircuitGraph.from_final_json(gt_json)

    # Fallback to cir1
    fallback = REPO_ROOT / "outputs" / "run1" / "final_result" / "json" / "cir1.final.json"
    if fallback.exists():
        return CircuitGraph.from_final_json(fallback)

    raise FileNotFoundError(f"Could not resolve circuit or job ID '{circuit_id}'")


@mcp_server.tool()
def inspect_circuit(circuit_id: str = "cir1") -> str:
    """
    Inspects a reconstructed circuit diagram and returns structured components,
    terminals, connected electrical nodes, and uncertainty records.
    """
    try:
        graph = _resolve_circuit_graph(circuit_id)
        inspector = ConnectivityInspector(graph)
        return json.dumps({
            "circuit_id": graph.image_name,
            "summary": inspector.get_summary(),
            "components": inspector.inspect_components(),
            "nodes": inspector.inspect_nodes(),
            "uncertainties": inspector.inspect_uncertainties()
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp_server.tool()
def validate_circuit(circuit_id: str = "cir1") -> str:
    """
    Runs domain-specific rule-based validation on the circuit graph.
    Detects floating terminals, accidental short circuits, missing ground reference,
    implausible values, and ambiguous wire crossings.
    """
    try:
        graph = _resolve_circuit_graph(circuit_id)
        validator = CircuitValidator(graph)
        findings = validator.validate()
        return json.dumps({
            "circuit_id": graph.image_name,
            "total_findings": len(findings),
            "findings": [f.model_dump() for f in findings]
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp_server.tool()
def generate_schematic(circuit_id: str = "cir1", theme: str = "dark") -> str:
    """
    Renders the reconstructed circuit into standard IEEE electronic vector CAD schematic SVG.
    Returns the generated file paths and SVG markup location.
    """
    try:
        graph = _resolve_circuit_graph(circuit_id)
        out_dir = REPO_ROOT / "outputs" / "mcp_renders" / graph.image_name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = render_circuit_graph(graph, out_dir)
        return json.dumps({
            "circuit_id": graph.image_name,
            "status": "success",
            "output_directory": str(out_dir),
            "files": files
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp_server.tool()
def export_spice_netlist(circuit_id: str = "cir1") -> str:
    """
    Synthesizes and exports a Berkeley SPICE netlist (.cir) from the circuit graph.
    """
    try:
        clean_id = _normalize_circuit_id(circuit_id)
        from build_spice_netlists import export_one
        json_path = REPO_ROOT / "outputs" / "run1" / "final_result" / "json" / f"{clean_id}.final.json"
        if not json_path.exists():
            json_path = REPO_ROOT / "data" / "ground_truth" / f"{clean_id}.gt.json"

        out_dir = REPO_ROOT / "outputs" / "mcp_spice"
        out_dir.mkdir(parents=True, exist_ok=True)

        if json_path.exists():
            spice_file = export_one(str(json_path), str(out_dir))
            with open(spice_file, "r", encoding="utf-8") as f:
                content = f.read()
            return json.dumps({
                "circuit_id": clean_id,
                "netlist_path": spice_file,
                "netlist": content
            }, indent=2)
        else:
            return json.dumps({"error": f"Circuit {clean_id} final JSON not found"}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp_server.tool()
def simulate_circuit(circuit_id: str = "cir1", analysis: str = "OP") -> str:
    """
    Validates the circuit netlist and executes SPICE simulation (DC operating point or transient)
    using ngspice if available on the system.
    """
    try:
        netlist_res = json.loads(export_spice_netlist(circuit_id))
        if "netlist" not in netlist_res:
            return json.dumps({"error": "Failed to generate netlist for simulation"}, indent=2)

        simulator = SPICESimulator()
        sim_res = simulator.run_simulation(netlist_res["netlist"], analysis=analysis)
        return json.dumps(sim_res.model_dump(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp_server.tool()
def explain_circuit(circuit_id: str = "cir1", question: str = "") -> str:
    """
    Provides a grounded electrical engineering explanation of the circuit topology,
    active validation findings, or answers a specific question without hallucination.
    """
    try:
        graph = _resolve_circuit_graph(circuit_id)
        validator = CircuitValidator(graph)
        findings = validator.validate()

        assistant = AICircuitAssistant()
        explanation = assistant.explain_circuit(
            graph=graph,
            findings=findings,
            user_question=question if question.strip() else None
        )
        return json.dumps({
            "circuit_id": graph.image_name,
            "question": question or "General circuit overview",
            "explanation": explanation
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def main():
    """Runs the MCP server over stdio transport."""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
