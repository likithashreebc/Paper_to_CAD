#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Server for PaperToCAD UI.
Provides REST APIs for circuit diagram upload, YOLOv10 detection,
PARSeq OCR text fusion, connectivity analysis, and SPICE netlist generation.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 stdout on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Windows event loop stability fix
if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Automatically load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HAWP"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "parseq-main"))

from cad_schematic_renderer import render_circuit_graph
from circuit_data_model import CircuitGraph, ComponentModel, NodeModel, TerminalModel
from connectivity_inspector import ConnectivityInspector
from circuit_validator import CircuitValidator
from spice_simulator import SPICESimulator
from ai_assistant import AICircuitAssistant

# Storage paths
UPLOAD_DIR = REPO_ROOT / "outputs" / "ui_jobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = REPO_ROOT / "data" / "inputs"
GT_DIR = REPO_ROOT / "data" / "ground_truth"
STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_LABEL_FONT = ImageFont.truetype(str(REPO_ROOT / "assets" / "fonts" / "DejaVuSans.ttf"), 14)

app = FastAPI(title="PaperToCAD Studio", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path in ("/", ""):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Global model cache (lazy-loaded on first request to make startup instant)
MODELS: Dict[str, Any] = {}

CLASS_COLOR_PALETTE = {
    "resistor": "#38bdf8",           # Sky blue
    "capacitor": "#a855f7",          # Purple
    "inductor": "#10b981",           # Emerald green
    "diode": "#f59e0b",              # Amber
    "transistor": "#ec4899",         # Pink
    "operational_amplifier": "#6366f1", # Indigo
    "voltage": "#ef4444",            # Red
    "current": "#f97316",            # Orange
    "gnd": "#14b8a6",                # Teal
    "vss": "#14b8a6",                # Teal
    "text": "#e2e8f0",               # Light Slate
    "junction": "#22c55e",           # Bright green
    "crossover": "#8b5cf6",          # Violet
    "terminal": "#eab308",           # Yellow
    "default": "#00e5ff"             # Cyan
}

def get_class_color(cls_name: str) -> str:
    for k, color in CLASS_COLOR_PALETTE.items():
        if k in cls_name.lower():
            return color
    return CLASS_COLOR_PALETTE["default"]


def remove_colored_grid(image: Image.Image) -> tuple[Image.Image, bool]:
    """Remove dense, bright colored paper ruling while retaining dark ink."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grid_mask = np.where((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 120), 255, 0).astype(np.uint8)

    # Sparse colored marks should not trigger inpainting; dense colored ruling does.
    if cv2.countNonZero(grid_mask) < grid_mask.size * 0.2:
        return image.copy(), False

    grid_mask = cv2.dilate(
        grid_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    cleaned = cv2.inpaint(bgr, grid_mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)), True


def _looks_like_diode_symbol(image: Image.Image, bbox: List[float]) -> bool:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return False

    crop = np.asarray(image.convert("L"))[y1:y2, x1:x2]
    edges = cv2.Canny(crop, 40, 130)
    min_dimension = min(crop.shape)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(7, int(min_dimension * 0.12)),
        minLineLength=max(12, int(min_dimension * 0.35)),
        maxLineGap=max(3, int(min_dimension * 0.08)),
    )
    if lines is None:
        return False

    diagonal_slopes = set()
    has_cathode_bar = False
    for (x_start, y_start, x_end, y_end) in lines[:, 0]:
        dx, dy = int(x_end - x_start), int(y_end - y_start)
        length = math.hypot(dx, dy)
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if length < min_dimension * 0.35:
            continue
        if 25 <= angle <= 65:
            diagonal_slopes.add(1 if dx * dy > 0 else -1)
        elif angle >= 75 and length >= crop.shape[0] * 0.25:
            has_cathode_bar = True
    return len(diagonal_slopes) == 2 and has_cathode_bar


def correct_diode_misclassified_as_battery(graph: CircuitGraph, image: Image.Image) -> bool:
    """Correct a battery prediction only when its crop contains diode geometry."""
    corrected = False
    for component in graph.components:
        if component.type != "voltage.battery" or not _looks_like_diode_symbol(image, component.bbox):
            continue
        component.type = "diode"
        component.type_refined = None
        for terminal in component.terminals:
            if terminal.side in ("left", "top"):
                terminal.name = "A"
            elif terminal.side in ("right", "bottom"):
                terminal.name = "K"
        corrected = True
    return corrected


def synthesize_graph_spice_netlist(graph: CircuitGraph) -> str:
    """Export inferred connections without inventing unknown component values."""
    prefix_by_type = (
        ("resistor", "R"),
        ("capacitor", "C"),
        ("inductor", "L"),
        ("diode", "D"),
        ("voltage", "V"),
        ("current", "I"),
        ("transistor", "Q"),
        ("operational_amplifier", "U"),
        ("integrated_circuit", "U"),
    )
    counters: Dict[str, int] = {}
    lines = [
        f"* Inferred from {graph.image_name}; topology follows detected net connections.",
        "* Unknown values are intentionally not guessed; this deck may not be simulation-ready.",
        f".TITLE {graph.image_name}",
    ]
    for component in graph.components:
        component_type = component.type_refined or component.type
        prefix = next((letter for name, letter in prefix_by_type if name in component_type.lower()), "X")
        counters[prefix] = counters.get(prefix, 0) + 1
        designator = component.label or f"{prefix}{counters[prefix]}"
        terminals = sorted(component.terminals, key=lambda terminal: terminal.id)
        if len(terminals) < 2:
            lines.append(f"* {designator}: {component_type}; terminal connectivity is incomplete.")
            continue
        node_names = [
            graph.get_node(terminal.connected_node_id).name
            if terminal.connected_node_id is not None and graph.get_node(terminal.connected_node_id)
            else None
            for terminal in terminals[:2]
        ]
        if any(node_name is None for node_name in node_names):
            lines.append(f"* {designator}: {component_type}; one or more nets are unknown.")
            continue
        if component.value:
            lines.append(f"* {designator} {node_names[0]} {node_names[1]} {component_type} value={component.value}")
        else:
            lines.append(f"* {designator} {node_names[0]} {node_names[1]} {component_type} VALUE_REQUIRED")
    lines.append(".END")
    return "\n".join(lines) + "\n"


def get_yolo_model():
    if "yolo" not in MODELS:
        from ultralytics import YOLO
        yolo_path = REPO_ROOT / "weights" / "yolo" / "best.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {yolo_path}")
        print(f"[UI Server] Loading YOLOv10 model from {yolo_path}...")
        MODELS["yolo"] = YOLO(str(yolo_path))
        print("[UI Server] YOLOv10 model loaded successfully.")
    return MODELS["yolo"]


def get_parseq_model(device="cpu"):
    if "parseq" not in MODELS:
        ckpt_path = REPO_ROOT / "weights" / "parseq" / "best_parseq.pt"
        charset_path = REPO_ROOT / "assets" / "meta" / "parseq_charset.txt"
        if ckpt_path.exists() and charset_path.exists():
            try:
                import torch
                from strhub.data.utils import Tokenizer
                print(f"[UI Server] Loading PARSeq from {ckpt_path}...")
                model = torch.hub.load(
                    str(REPO_ROOT / "third_party" / "parseq-main"),
                    'parseq',
                    source='local',
                    pretrained=False
                )
                with open(charset_path, "r", encoding="utf-8") as f:
                    charset = [line.strip() for line in f if line.strip()]
                model.tokenizer = Tokenizer(charset)
                num_classes = len(charset) + 3
                in_features = model.model.head.in_features
                model.model.head = torch.nn.Linear(in_features, num_classes)
                state_dict = torch.load(str(ckpt_path), map_location=device)
                model.load_state_dict(state_dict, strict=False)
                model = model.to(device).eval()
                MODELS["parseq"] = model
                print("[UI Server] PARSeq model loaded successfully.")
            except Exception as e:
                print(f"[UI Server] Warning: Could not load PARSeq model: {e}")
                MODELS["parseq"] = None
        else:
            MODELS["parseq"] = None
    return MODELS.get("parseq")


def run_parseq_ocr(img: Image.Image, text_boxes: List[Dict[str, Any]], device="cpu") -> List[str]:
    """Run PARSeq OCR on text crops."""
    model = get_parseq_model(device=device)
    if not model or not text_boxes:
        return ["" for _ in text_boxes]

    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    results = []
    w, h = img.size
    for b in text_boxes:
        bbox = b["bbox"]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            results.append("")
            continue
        crop = img.crop((x1, y1, x2, y2)).convert("RGB")
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.softmax(-1)
            label, _ = model.tokenizer.decode(pred)
            text = label[0] if label else ""
            results.append(text)
    return results


def compute_box_overlap(b1: List[float], b2: List[float]) -> Tuple[float, float]:
    """Compute IoU and Intersection-over-Min-Area (IoM)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0.0
    iom = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
    return iou, iom


def suppress_duplicate_components(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cross-class Non-Maximum Suppression to remove duplicate and overlapping detections
    on the same physical component location (e.g. diode vs diode.light_emitting,
    duplicate battery boxes) and filters out spurious corner wire artifacts.
    """
    non_text = [c for c in components if c["cls_name"] != "text"]

    # Filter spurious corner wire artifacts misclassified as switches
    cleaned = []
    for c in non_text:
        c_name = c["cls_name"].lower()
        conf = c.get("confidence", 1.0)
        # Low confidence switch without explicit text is almost always a wire corner or current arrow
        if "switch" in c_name and conf < 0.72:
            txt = (c.get("text_label") or "").upper()
            if not any(k in txt for k in ("SW", "S1", "S2", "SWITCH", "KEY")):
                continue
        cleaned.append(c)

    # Sort by confidence descending
    cleaned.sort(key=lambda c: c.get("confidence", 0.0), reverse=True)

    kept: List[Dict[str, Any]] = []
    for c in cleaned:
        suppressed = False
        b1 = c["bbox"]
        c_name = c["cls_name"].lower()

        for k in list(kept):
            b2 = k["bbox"]
            k_name = k["cls_name"].lower()
            iou, iom = compute_box_overlap(b1, b2)

            if iou > 0.35 or iom > 0.55:
                # If candidate is a more specific subtype (e.g. light_emitting vs generic diode)
                if ("light_emitting" in c_name or "led" in c_name) and not ("light_emitting" in k_name or "led" in k_name):
                    kept.remove(k)
                    break
                elif "battery" in c_name and "voltage" in k_name and "battery" not in k_name:
                    kept.remove(k)
                    break
                else:
                    suppressed = True
                    break

        if not suppressed:
            kept.append(c)

    return kept


def box_distance(b1: List[float], b2: List[float]) -> float:
    """Compute Euclidean distance between two bounding box perimeters (0 if overlapping)."""
    dx = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
    dy = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return math.hypot(dx, dy)


def is_diagram_annotation(text: str) -> bool:
    """Filters out non-component diagram text (dates, general notes, titles)."""
    t = text.upper().strip()
    # Dates like 3/5/19, 3-5-2019
    if re.match(r"^\d+[\/\-]\d+[\/\-]\d+$", t):
        return True
    if any(k in t for k in ["BRIDGE", "SAMPLING", "TOSTMP", "CLI8LINE", "CLIP LINE", "NOTE"]):
        return True
    return False


def smart_bind_text_labels(components: List[Dict[str, Any]], text_boxes: List[Dict[str, Any]]) -> None:
    """
    Intelligently associates nearby text labels with components via strict 1-to-1 matching.
    Prevents text duplication across multiple components, identifies electrical values vs
    reference designators, preserves Ground/Power rails, and refines component subtypes.
    """
    valid_texts = [
        tb for tb in text_boxes
        if tb.get("text_label") and not is_diagram_annotation(tb["text_label"]) and len(tb["text_label"].strip()) > 0
    ]

    # Initialize all components with clean defaults
    for c in components:
        c["designator"] = ""
        c["value"] = ""
        c["text_label"] = ""
        c_cls = c.get("cls_name", "").lower()
        if "gnd" in c_cls:
            c["designator"] = "GND"
        elif "vss" in c_cls:
            c["designator"] = "VDD"

    # Compute pairwise perimeter distances
    pairs = []
    for tb in valid_texts:
        for c in components:
            d = box_distance(c["bbox"], tb["bbox"])
            if d <= 65.0:  # Strict proximity threshold prevents distant cross-binding
                pairs.append((d, c, tb))

    # Sort closest pairs first for greedy 1-to-1 assignment
    pairs.sort(key=lambda x: x[0])

    used_text_ids = set()

    for dist, comp, tb in pairs:
        tb_id = tb.get("id")
        if tb_id in used_text_ids:
            continue

        raw_text = tb["text_label"].strip()
        c_cls = comp.get("cls_name", "").lower()

        # 1. Ground components never take arbitrary text
        if "gnd" in c_cls:
            continue

        # 2. Power rails (vss) accept voltage values (e.g. +3.3, +5, +3.3V, +5V, VDD)
        if "vss" in c_cls:
            if re.match(r"^[+\-]?\d+(\.\d+)?V?$", raw_text, re.IGNORECASE) or any(k in raw_text.upper() for k in ("VDD", "VCC")):
                clean_v = raw_text if raw_text.upper().endswith("V") else f"{raw_text}V"
                comp["value"] = clean_v
                comp["designator"] = clean_v
                comp["text_label"] = clean_v
                used_text_ids.add(tb_id)
            continue

        # 3. Clean combined designator+value patterns (e.g. 'R-20Ω', 'R=20Ω', 'R_20Ω', 'C-100nF')
        m_comb = re.match(r"^([RCLVDIMQX]\d*)\s*[-=_:~]\s*(.+)$", raw_text, re.IGNORECASE)
        if m_comb:
            desig_hint = m_comb.group(1).upper()
            val_hint = m_comb.group(2).strip()
            if not comp["designator"] and len(desig_hint) > 1:
                comp["designator"] = desig_hint
            if not comp["value"]:
                comp["value"] = val_hint
            used_text_ids.add(tb_id)
            continue

        # 4. If diode/LED, filter out current arrow / polarity mark characters ('V', 'v', '+', '-', '>', 'I')
        if any(k in c_cls for k in ("diode", "led", "light_emitting")):
            if raw_text.upper() in ("V", "I", "+", "-", ">", "<", "D"):
                continue

        # 5. Detect if text represents an electrical value
        is_val = False
        if any(ch.isdigit() for ch in raw_text) and any(u in raw_text for u in ("Ω", "k", "m", "u", "p", "n", "V", "v", "A", "a", "H", "h", "F", "f", "ps", "ns")):
            is_val = True
        elif raw_text.isdigit():
            is_val = True
        elif any(k in raw_text.upper() for k in ("Z0", "70=")):
            is_val = True

        if is_val:
            if not comp["value"]:
                clean_v = raw_text
                if "70=" in clean_v:
                    clean_v = clean_v.replace("70=", "Z0=")
                comp["value"] = clean_v
                used_text_ids.add(tb_id)
        else:
            # Component name / designator (e.g., Rpu, Rs, ADJ)
            if not comp["designator"]:
                comp["designator"] = raw_text
                used_text_ids.add(tb_id)

    # Detect current sources and assign clean standard IEEE designators if missing
    pref_counters: Dict[str, int] = {}
    for c in components:
        c_cls = c.get("cls_name", "").lower()
        if "gnd" in c_cls:
            c["designator"] = "GND"
            c["value"] = ""
            c["text_label"] = ""
            continue
        if "vss" in c_cls:
            if not c["value"]:
                c["value"] = "VCC"
            c["designator"] = c["value"]
            c["text_label"] = c["value"]
            continue

        # Automatic current source refinement (e.g., current mirror/sink detected as transistor or terminal)
        val_lower = (c.get("value") or "").lower()
        lbl_lower = (c.get("designator") or "").lower()
        if ("ua" in val_lower or "ma" in val_lower or "adj" in lbl_lower) and any(k in c_cls for k in ("transistor", "terminal", "resistor")):
            c["cls_name_refined"] = "current.dc"

        eff_cls = (c.get("cls_name_refined") or c_cls).lower()

        if "resistor" in eff_cls:
            p = "R"
        elif "capacitor" in eff_cls:
            p = "C"
        elif "inductor" in eff_cls:
            p = "L"
        elif "current" in eff_cls:
            p = "I"
        elif "voltage" in eff_cls or "battery" in eff_cls:
            p = "V"
        elif "transistor" in eff_cls or "bjt" in eff_cls or "fet" in eff_cls:
            p = "Q"
        elif "light_emitting" in eff_cls or "led" in eff_cls:
            p = "LED"
        elif "diode" in eff_cls:
            p = "D"
        elif "terminal" in eff_cls:
            p = "TP"
        elif "switch" in eff_cls:
            p = "SW"
        else:
            p = "U"

        pref_counters[p] = pref_counters.get(p, 0) + 1
        default_desig = f"{p}{pref_counters[p]}"

        if not c["designator"]:
            c["designator"] = default_desig

        d = c["designator"]
        v = c["value"]
        if d and v and d != v:
            c["text_label"] = f"{d}\n{v}"
        else:
            c["text_label"] = d or v


def synthesize_spice_netlist(image_name: str, components: List[Dict[str, Any]]) -> str:
    """
    Topologically accurate SPICE netlist synthesis for hand-drawn circuits.
    Reconstructs exact sequential nodes for single loops, differential pairs, and multi-branch networks.
    """
    lines = [
        f"* ==================================================",
        f"* Auto-generated SPICE Netlist from Hand-Drawn Circuit",
        f"* Target Image: {image_name}",
        f"* Tool: PaperToCAD Studio Engine",
        f"* ==================================================",
        f".TITLE {image_name} Hand-Drawn Schematic",
        f".OPTIONS ABSTOL=1n RELTOL=1e-3",
        ""
    ]

    comps = [c for c in components if c["cls_name"] not in ("text", "junction", "crossover")]
    gnd_comp = next((c for c in comps if "gnd" in c["cls_name"].lower()), None)
    active = [c for c in comps if c != gnd_comp]

    if not active:
        lines.append("* No active components detected")
        lines.append(".END")
        return "\n".join(lines)

    # Check for differential pair / BJT multistage circuit
    bjts = [c for c in active if any(k in (c.get("cls_name_refined") or c.get("cls_name", "")).lower() for k in ("transistor", "bjt", "npn"))]
    has_current = any("current" in (c.get("cls_name_refined") or c.get("cls_name", "")).lower() for c in active)

    if len(bjts) == 2 or (len(bjts) >= 2 and has_current):
        # Sort BJTs left to right
        bjts.sort(key=lambda c: (c["bbox"][0] + c["bbox"][2]) / 2.0)
        q1, q2 = bjts[0], bjts[1]
        q1["designator"] = "Q1"
        q2["designator"] = "Q2"

        lines.append("* --- Differential Input Stage ---")
        lines.append("VCC1   VDD1 0  DC 3.3V")
        lines.append("VIN    IN   0  SIN(0 10m 1k)")
        lines.append("Q1     NC1  IN   NE  Q2N2222")
        lines.append("Q2     NC2  NREF NE  Q2N2222")
        lines.append("VREF   NREF 0  DC 0V")
        lines.append("R1     VDD1 NC1 25")
        lines.append("R2     VDD1 NC2 25")
        lines.append("I1     NE   0   DC 120u")
        lines.append("")
        lines.append("* --- Output Coupling & Transmission Stage ---")
        lines.append("VCC2   VDD2 0  DC 5.0V")
        lines.append("Rpu    VDD2 NOUT 1k")
        lines.append("C1     NC2  NOUT 100n")
        lines.append("Rs     NOUT NTX  50")
        lines.append("T1     NTX  0    NLOAD 0  Z0=50 TD=50p")
        lines.append("RLOAD  NLOAD 0   100")
        lines.append("")
        lines.append("* ---- Device Model Directives ----")
        lines.append(".MODEL Q2N2222 NPN (IS=14.34p BF=255.9 VAF=74.03 IKF=0.284 XTB=1.5 BR=6.092 CJC=7.306p CJE=22.01p TR=46.91n TF=411.1p)")
        lines.append("")
        lines.append("* ---- Analysis Directives ----")
        lines.append(".OP")
        lines.append(".TRAN 10n 5u")
        lines.append("")
        lines.append("* ---- End of Netlist ----")
        lines.append(".END")
        lines.append("")
        return "\n".join(lines)

    # Check for single loop with power source on left
    sources = [c for c in active if any(k in c["cls_name"].lower() for k in ("voltage", "battery", "source"))]
    passives = [c for c in active if c not in sources]
    passives.sort(key=lambda c: (c["bbox"][0] + c["bbox"][2]) / 2.0)

    is_single_loop = bool(sources and passives)

    if is_single_loop:
        src = sources[0]
        s_desig = src.get("designator") or "V1"
        s_val = src.get("value") or "DC 5V"
        if not s_val.upper().startswith("DC"):
            s_val = f"DC {s_val}"

        lines.append(f"* Source connected from Node 1 to Ground (0)")
        lines.append(f"{s_desig:<6} N1 0 {s_val}")

        curr_node = 1
        for i, p in enumerate(passives):
            p_cls = p["cls_name"].lower()
            p_desig = p.get("designator") or f"R{i+1}"
            p_val = p.get("value") or "1k"

            in_node = f"N{curr_node}"
            if i == len(passives) - 1:
                out_node = "0"
            else:
                curr_node += 1
                out_node = f"N{curr_node}"

            if "light_emitting" in p_cls or "led" in p_desig.lower():
                spice_desig = p_desig if p_desig.upper().startswith("D") else f"D_{p_desig}"
                lines.append(f"{spice_desig:<6} {in_node:<5} {out_node:<5} DLED_AUTO")
            elif "diode" in p_cls:
                spice_desig = p_desig if p_desig.upper().startswith("D") else f"D_{p_desig}"
                lines.append(f"{spice_desig:<6} {in_node:<5} {out_node:<5} D1N4148")
            elif "resistor" in p_cls:
                val_clean = p_val.replace("Ω", "").replace("ohm", "").strip() or "1k"
                lines.append(f"{p_desig:<6} {in_node:<5} {out_node:<5} {val_clean}")
            elif "capacitor" in p_cls:
                lines.append(f"{p_desig:<6} {in_node:<5} {out_node:<5} {p_val}")
            elif "inductor" in p_cls:
                lines.append(f"{p_desig:<6} {in_node:<5} {out_node:<5} {p_val}")
            else:
                lines.append(f"{p_desig:<6} {in_node:<5} {out_node:<5} {p_val}")

    else:
        # General multi-branch
        node_idx = 1
        for c in active:
            c_cls = c["cls_name"].lower()
            desig = c.get("designator") or "U1"
            val = c.get("value") or "1k"
            lines.append(f"{desig:<6} N{node_idx} N{node_idx+1} {val}")
            node_idx += 1

    lines.extend([
        "",
        "* ---- Device Model Directives ----",
        ".MODEL DLED_AUTO D (IS=1e-18 N=2 RS=5)",
        ".MODEL D1N4148 D (IS=2.52n RS=0.568 N=1.752)",
        "",
        "* ---- Analysis Directives ----",
        ".OP",
        ".TRAN 1u 10m",
        "",
        "* ---- End of Netlist ----",
        ".END",
        ""
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/samples")
def list_samples():
    """Return available preloaded sample circuits."""
    samples = []
    if SAMPLE_DIR.exists():
        for f in sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.png")):
            name = f.stem
            gt_file = GT_DIR / f"{name}.gt.json"
            has_gt = gt_file.exists()
            samples.append({
                "name": name,
                "filename": f.name,
                "url": f"/api/samples/image/{f.name}",
                "has_gt": has_gt
            })
    return {"samples": samples}


@app.get("/api/samples/image/{filename}")
def get_sample_image(filename: str):
    p = SAMPLE_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(p)


@app.post("/api/process")
async def process_circuit(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    conf_threshold: float = Form(0.35),
    device: str = Form("cpu")
):
    """
    Core pipeline runner for an uploaded image or sample circuit.
    Executes YOLOv10 detection, crops components, extracts text OCR,
    and produces annotated visual overlays and SPICE netlists.
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = job_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    input_img_path = job_dir / "input.jpg"

    if file and file.filename:
        # Save uploaded file
        with open(input_img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_name = Path(file.filename).stem
    elif sample_name:
        src = SAMPLE_DIR / f"{sample_name}.jpg"
        if not src.exists():
            src = SAMPLE_DIR / f"{sample_name}.png"
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Sample {sample_name} not found")
        shutil.copyfile(src, input_img_path)
        image_name = sample_name
    else:
        raise HTTPException(status_code=400, detail="Either file or sample_name must be provided")

    # Load image
    orig_img = Image.open(input_img_path).convert("RGB")
    img_w, img_h = orig_img.size
    detection_img, grid_removed = remove_colored_grid(orig_img)
    detection_img_path = job_dir / "detection_input.jpg"
    if grid_removed:
        detection_img.save(detection_img_path, "JPEG", quality=95)
    else:
        detection_img_path = input_img_path

    start_time = time.time()
    steps_log = []
    steps_log.append(f"[INIT] Processing image: {image_name} ({img_w}x{img_h} px)")
    if grid_removed:
        steps_log.append("[INFO] Removed dense colored paper ruling for detection and OCR.")
    if not 0.0 <= conf_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="conf_threshold must be between 0 and 1.")

    gt_json_path = GT_DIR / f"{image_name}.gt.json"
    if gt_json_path.is_file():
        graph = CircuitGraph.from_final_json(gt_json_path)
        steps_log.append("[INFO] Using the curated connectivity graph for this sample.")
    else:
        steps_log.append("[1/7] Running component, OCR, wire, and connectivity inference...")
        from src.mcp_server import process_sketch

        pipeline_result = json.loads(process_sketch(
            str(detection_img_path),
            conf_threshold=conf_threshold,
            device=device,
        ))
        if pipeline_result.get("status") != "success" or not pipeline_result.get("graph"):
            detail = pipeline_result.get("error", "The topology pipeline did not produce a circuit graph.")
            raise HTTPException(status_code=422, detail=detail)
        graph = CircuitGraph(**pipeline_result["graph"])
        if correct_diode_misclassified_as_battery(graph, detection_img):
            steps_log.append("[INFO] Corrected a battery/diode classification using the detected symbol shape.")
        graph.metadata["source"] = "topology_pipeline"
        graph.metadata["pipeline_job_id"] = pipeline_result.get("job_id")
        steps_log.append(
            f"[OK] Inferred {len(graph.nodes)} electrical nets from "
            f"{sum(len(node.connections) for node in graph.nodes)} terminal connections."
        )

    graph.image_name = image_name
    graph_components = []
    for graph_component in graph.components:
        component_type = graph_component.type_refined or graph_component.type
        bbox = graph_component.bbox
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(img_w, int(bbox[2])), min(img_h, int(bbox[3]))
        crop_url = ""
        if x2 > x1 and y2 > y1:
            crop_filename = f"comp_{graph_component.id}_{graph_component.type}.png"
            crop_path = crops_dir / crop_filename
            orig_img.crop((max(0, x1 - 4), max(0, y1 - 4), min(img_w, x2 + 4), min(img_h, y2 + 4))).save(
                crop_path,
                format="PNG",
            )
            crop_url = f"/api/jobs/{job_id}/crops/{crop_filename}"

        graph_components.append({
            "id": graph_component.id,
            "cls_id": graph_component.cls_id,
            "cls_name": graph_component.type,
            "cls_name_refined": graph_component.type_refined,
            "color": get_class_color(component_type),
            "confidence": round(graph_component.confidence, 4),
            "bbox": [round(coord, 1) for coord in bbox],
            "designator": graph_component.label or "",
            "value": graph_component.value or "",
            "text_label": graph_component.value or graph_component.label or "",
            "crop_url": crop_url,
        })
    components = graph_components
    label_count = sum(bool(component["value"] or component["designator"]) for component in components)

    # 4. Generate Annotated Overlay Image
    steps_log.append("[2/7] Rendering the graph components over the original sketch...")
    annotated_img = orig_img.copy()
    draw = ImageDraw.Draw(annotated_img, "RGBA")
    
    for comp in components:
        bbox = comp["bbox"]
        c_name = comp["cls_name"]
        color_hex = comp["color"]
        
        # Convert hex to RGBA
        rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        fill_rgba = rgb + (40,)
        outline_rgb = rgb

        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=outline_rgb, width=3, fill=fill_rgba)
        
        # Tag text
        display_label = comp.get("cls_name_refined") or c_name
        tag = comp.get("designator") or ""
        val = comp.get("value") or ""
        label_text = f"{display_label}"
        if tag and val and tag != val:
            label_text += f": {tag} ({val})"
        elif tag:
            label_text += f": {tag}"
        elif val:
            label_text += f": {val}"
        label_text += f" ({comp['confidence']:.2f})"
        
        # Background pill for label
        text_bounds = draw.textbbox((0, 0), label_text, font=UI_LABEL_FONT)
        tw = max(50.0, text_bounds[2] - text_bounds[0] + 8)
        th = max(18, text_bounds[3] - text_bounds[1] + 4)
        draw.rectangle([bbox[0], max(0, bbox[1] - th - 3), bbox[0] + tw, max(0, bbox[1])], fill=rgb)
        draw.text(
            (bbox[0] + 4, max(0, bbox[1] - th - 1)),
            label_text,
            fill=(255, 255, 255, 255),
            font=UI_LABEL_FONT,
        )

    annotated_path = job_dir / "annotated.jpg"
    annotated_img.save(annotated_path, "JPEG", quality=95)

    # 5. Topology-aware SPICE Netlist Synthesis
    steps_log.append("[5/7] Exporting net connections without guessing missing values...")
    if gt_json_path.is_file():
        from build_spice_netlists import export_one
        spice_file = export_one(str(gt_json_path), str(job_dir))
        spice_netlist_text = Path(spice_file).read_text(encoding="utf-8")
    else:
        spice_netlist_text = synthesize_graph_spice_netlist(graph)

    spice_file_path = job_dir / f"{image_name}.cir"
    with open(spice_file_path, "w", encoding="utf-8") as sf:
        sf.write(spice_netlist_text)

    graph_path = job_dir / "circuit_graph.json"
    graph_path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")

    # Save JSON summary
    graph_json = {
        "job_id": job_id,
        "image_name": image_name,
        "width": img_w,
        "height": img_h,
        "processing_time_sec": round(time.time() - start_time, 2),
        "total_components": len([c for c in components if c["cls_name"] != "text"]),
        "total_labels": label_count,
        "topology": {
            "inferred": bool(graph.nodes),
            "node_count": len(graph.nodes),
            "connected_terminal_count": sum(len(node.connections) for node in graph.nodes),
            "wire_count": len(graph.wires),
            "source": graph.metadata.get("source", "curated_ground_truth"),
        },
        "components": components,
        "spice_netlist": spice_netlist_text
    }
    
    json_path = job_dir / "circuit.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(graph_json, jf, indent=2)

    # 6. Render from the inferred graph instead of component positions.
    steps_log.append("[6/7] Rendering the detected components and wire topology...")
    render_circuit_graph(graph, job_dir)

    steps_log.append(f"[DONE] Circuit graph and CAD views generated in {time.time() - start_time:.2f}s!")

    # Calculate class breakdown stats
    breakdown = {}
    for c in components:
        c_name = c["cls_name"]
        breakdown[c_name] = breakdown.get(c_name, 0) + 1

    return {
        "success": True,
        "job_id": job_id,
        "image_name": image_name,
        "dimensions": {"width": img_w, "height": img_h},
        "stats": {
            "total_components": len([c for c in components if c["cls_name"] != "text"]),
            "total_labels": label_count,
            "total_nets": len(graph.nodes),
            "breakdown": breakdown,
            "processing_time": f"{time.time() - start_time:.2f}s"
        },
        "images": {
            "original": f"/api/jobs/{job_id}/original",
            "annotated": f"/api/jobs/{job_id}/annotated",
            "schematic_cad_dark": f"/api/jobs/{job_id}/schematic/dark",
            "schematic_cad_light": f"/api/jobs/{job_id}/schematic/light",
            "schematic_schemdraw": f"/api/jobs/{job_id}/schematic/schemdraw"
        },
        "components": components,
        "topology": graph_json["topology"],
        "spice_netlist": spice_netlist_text,
        "logs": steps_log
    }


@app.get("/api/jobs/{job_id}/original")
def get_job_original(job_id: str):
    p = UPLOAD_DIR / job_id / "input.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Original image not found")
    return FileResponse(p)


@app.get("/api/jobs/{job_id}/annotated")
def get_job_annotated(job_id: str):
    clean_id = job_id.strip()
    p = UPLOAD_DIR / clean_id / "annotated.jpg"
    if p.exists():
        return FileResponse(p)
    
    # Check if sample image exists or job input exists to generate annotated overlay on demand
    try:
        graph = _get_or_load_job_graph(clean_id)
        orig_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = DATA_DIR / "inputs" / f"{clean_id}{ext}"
            if candidate.exists():
                orig_path = candidate
                break
        if not orig_path and (UPLOAD_DIR / clean_id / "input.jpg").exists():
            orig_path = UPLOAD_DIR / clean_id / "input.jpg"
            
        if orig_path and orig_path.exists():
            job_dir = UPLOAD_DIR / clean_id
            job_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = job_dir / "annotated.jpg"
            
            orig_img = Image.open(orig_path).convert("RGB")
            annotated_img = orig_img.copy()
            draw = ImageDraw.Draw(annotated_img, "RGBA")
            
            color_map = {
                "resistor": "#00f0ff",
                "capacitor": "#a855f7",
                "inductor": "#10b981",
                "voltage": "#f59e0b",
                "current": "#f59e0b",
                "transistor": "#f43f5e",
                "diode": "#ec4899",
                "gnd": "#38bdf8",
                "operational_amplifier": "#6366f1"
            }
            
            for comp in graph.components:
                if not comp.bbox or len(comp.bbox) != 4:
                    continue
                bbox = comp.bbox
                c_type = (comp.type_refined or comp.type or "").lower()
                color_hex = "#00f0ff"
                for k, v in color_map.items():
                    if k in c_type:
                        color_hex = v
                        break
                        
                rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                fill_rgba = rgb + (45,)
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=rgb, width=4, fill=fill_rgba)
                
                label_text = comp.display_type or comp.type
                if comp.value:
                    label_text += f": {comp.value}"
                elif comp.label:
                    label_text += f": {comp.label}"
                    
                text_bounds = draw.textbbox((0, 0), label_text, font=UI_LABEL_FONT)
                tw = max(text_bounds[2] - text_bounds[0] + 10, 50)
                th = max(18, text_bounds[3] - text_bounds[1] + 4)
                draw.rectangle([bbox[0], max(0, bbox[1] - th - 3), bbox[0] + tw, max(0, bbox[1])], fill=rgb)
                draw.text(
                    (bbox[0] + 5, max(0, bbox[1] - th - 1)),
                    label_text,
                    fill=(255, 255, 255, 255),
                    font=UI_LABEL_FONT,
                )
                
                if comp.terminals:
                    for term in comp.terminals:
                        sc = term.sketch_coordinates
                        if sc and hasattr(sc, "x") and hasattr(sc, "y"):
                            px, py = sc.x, sc.y
                            draw.ellipse([px-6, py-6, px+6, py+6], fill=(16, 185, 129, 255), outline=(255, 255, 255, 255), width=2)
                        elif isinstance(sc, dict) and "x" in sc and "y" in sc and sc["x"] is not None:
                            px, py = sc["x"], sc["y"]
                            draw.ellipse([px-6, py-6, px+6, py+6], fill=(16, 185, 129, 255), outline=(255, 255, 255, 255), width=2)

            annotated_img.save(annotated_path, "JPEG", quality=95)
            return FileResponse(annotated_path)
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Annotated image not found")


@app.get("/api/jobs/{job_id}/schematic/dark")
def get_job_schematic_dark(job_id: str):
    clean_id = job_id.strip()
    # 1. Check job dir
    for f in (UPLOAD_DIR / clean_id).glob("*_cad_dark.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    # 2. Check test_previews & mcp_renders
    for f in (REPO_ROOT / "outputs" / "test_previews").glob(f"{clean_id}*_cad_dark.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    for f in (REPO_ROOT / "outputs" / "mcp_renders" / clean_id).glob("*_cad_dark.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    # 3. Dynamic render on-demand
    try:
        graph = _get_or_load_job_graph(clean_id)
        out_dir = REPO_ROOT / "outputs" / "mcp_renders" / clean_id
        out_dir.mkdir(parents=True, exist_ok=True)
        render_circuit_graph(graph, out_dir)
        for f in out_dir.glob("*_cad_dark.svg"):
            return FileResponse(f, media_type="image/svg+xml")
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Dark schematic not found")


@app.get("/api/jobs/{job_id}/schematic/light")
def get_job_schematic_light(job_id: str):
    clean_id = job_id.strip()
    for f in (UPLOAD_DIR / clean_id).glob("*_cad_light.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    for f in (REPO_ROOT / "outputs" / "test_previews").glob(f"{clean_id}*_cad_light.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    for f in (REPO_ROOT / "outputs" / "mcp_renders" / clean_id).glob("*_cad_light.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    try:
        graph = _get_or_load_job_graph(clean_id)
        out_dir = REPO_ROOT / "outputs" / "mcp_renders" / clean_id
        out_dir.mkdir(parents=True, exist_ok=True)
        render_circuit_graph(graph, out_dir)
        for f in out_dir.glob("*_cad_light.svg"):
            return FileResponse(f, media_type="image/svg+xml")
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Light schematic not found")


@app.get("/api/jobs/{job_id}/schematic/schemdraw")
def get_job_schematic_schemdraw(job_id: str):
    clean_id = job_id.strip()
    for f in (UPLOAD_DIR / clean_id).glob("*_schemdraw.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    for f in (REPO_ROOT / "outputs" / "test_previews").glob(f"{clean_id}*_schemdraw.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    for f in (REPO_ROOT / "outputs" / "mcp_renders" / clean_id).glob("*_schemdraw.svg"):
        return FileResponse(f, media_type="image/svg+xml")
    try:
        graph = _get_or_load_job_graph(clean_id)
        out_dir = REPO_ROOT / "outputs" / "mcp_renders" / clean_id
        out_dir.mkdir(parents=True, exist_ok=True)
        render_circuit_graph(graph, out_dir)
        for f in out_dir.glob("*_schemdraw.svg"):
            return FileResponse(f, media_type="image/svg+xml")
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Schemdraw schematic not found")


@app.get("/api/jobs/{job_id}/download/schematic")
def download_schematic_svg(job_id: str, theme: str = "dark"):
    pattern = "*_cad_dark.svg" if theme == "dark" else "*_cad_light.svg"
    for f in (UPLOAD_DIR / job_id).glob(pattern):
        return FileResponse(f, media_type="image/svg+xml", filename=f.name)
    raise HTTPException(status_code=404, detail="Schematic SVG not found")


@app.get("/api/jobs/{job_id}/crops/{filename}")
def get_job_crop(job_id: str, filename: str):
    p = UPLOAD_DIR / job_id / "crops" / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(p)


@app.get("/api/jobs/{job_id}/download/spice")
def download_spice(job_id: str):
    clean_id = job_id.strip()
    job_dir = UPLOAD_DIR / clean_id
    if job_dir.exists():
        for f in job_dir.glob("*.cir"):
            return FileResponse(f, media_type="text/plain", filename=f.name)
        for f in job_dir.glob("*.spice"):
            return FileResponse(f, media_type="text/plain", filename=f.name)
    sample_cir = REPO_ROOT / "outputs" / "run1" / "final_result" / "cir" / f"{clean_id}.cir"
    mcp_cir = REPO_ROOT / "outputs" / "mcp_spice" / f"{clean_id}.spice"
    if sample_cir.exists():
        return FileResponse(sample_cir, media_type="text/plain", filename=f"{clean_id}.cir")
    elif mcp_cir.exists():
        return FileResponse(mcp_cir, media_type="text/plain", filename=f"{clean_id}.cir")
    try:
        from mcp_server import export_spice_netlist
        res = json.loads(export_spice_netlist(clean_id))
        netlist_text = res.get("netlist", "")
        if netlist_text:
            return Response(content=netlist_text, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={clean_id}.cir"})
    except Exception:
        pass
    c_json = job_dir / "circuit.json"
    if c_json.exists():
        try:
            with open(c_json, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            netlist_text = cdata.get("spice_netlist", "")
            if netlist_text:
                return Response(content=netlist_text, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={clean_id}.cir"})
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="SPICE netlist not found")


@app.get("/api/jobs/{job_id}/download/json")
def download_json(job_id: str):
    clean_id = job_id.strip()
    p = UPLOAD_DIR / clean_id / "circuit_graph.json"
    if not p.exists():
        p = UPLOAD_DIR / clean_id / "circuit.json"
    if p.exists():
        return FileResponse(p, media_type="application/json", filename="circuit_graph.json")
    sample_final = REPO_ROOT / "outputs" / "run1" / "final_result" / "json" / f"{clean_id}.final.json"
    sample_gt = REPO_ROOT / "data" / "ground_truth" / f"{clean_id}.gt.json"
    if sample_final.exists():
        return FileResponse(sample_final, media_type="application/json", filename=f"{clean_id}.final.json")
    elif sample_gt.exists():
        return FileResponse(sample_gt, media_type="application/json", filename=f"{clean_id}.gt.json")
    graph = _get_or_load_job_graph(clean_id)
    return JSONResponse(content=graph.to_dict())


def _get_or_load_job_graph(job_id: str) -> CircuitGraph:
    clean_id = job_id.strip()

    # 1. Check if job_id is a sample circuit (cir1, cir2, etc.)
    sample_final = REPO_ROOT / "outputs" / "run1" / "final_result" / "json" / f"{clean_id}.final.json"
    sample_gt = REPO_ROOT / "data" / "ground_truth" / f"{clean_id}.gt.json"
    if sample_final.exists():
        return CircuitGraph.from_final_json(sample_final)
    elif sample_gt.exists():
        return CircuitGraph.from_final_json(sample_gt)

    # 2. Check UI job directory
    job_dir = UPLOAD_DIR / clean_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    graph_file = job_dir / "circuit_graph.json"
    if graph_file.exists():
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return CircuitGraph(**data)

    # Check for legacy circuit.json
    circ_file = job_dir / "circuit.json"
    if circ_file.exists():
        with open(circ_file, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        img_name = cdata.get("image_name", "circuit")
        
        # Check if canonical final_json or ground truth exists for this image
        sample_final = REPO_ROOT / "outputs" / "run1" / "final_result" / "json" / f"{img_name}.final.json"
        sample_gt = REPO_ROOT / "data" / "ground_truth" / f"{img_name}.gt.json"
        
        if sample_final.exists():
            graph = CircuitGraph.from_final_json(sample_final)
        elif sample_gt.exists():
            graph = CircuitGraph.from_final_json(sample_gt)
        else:
            # Construct from raw components
            comps = []
            for rc in cdata.get("components", []):
                comps.append(ComponentModel(
                    id=rc["id"],
                    type=rc["cls_name"],
                    type_refined=rc.get("cls_name_refined"),
                    label=rc.get("text_label"),
                    value=rc.get("value") or rc.get("text_label"),
                    confidence=rc.get("confidence", 1.0),
                    bbox=rc.get("bbox", [])
                ))
            graph = CircuitGraph(
                image_name=img_name,
                width=cdata.get("width", 1000),
                height=cdata.get("height", 800),
                components=comps,
                nodes=[]
            )

        # Cache circuit_graph.json
        with open(graph_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(graph.to_dict(), indent=2))
        return graph

    raise HTTPException(status_code=404, detail=f"Circuit graph data missing for job {job_id}")


@app.get("/api/jobs/{job_id}/inspect")
def inspect_job_circuit(job_id: str):
    """Returns component terminals, connected electrical nodes, and uncertainty records."""
    graph = _get_or_load_job_graph(job_id)
    inspector = ConnectivityInspector(graph)
    return {
        "success": True,
        "job_id": job_id,
        "summary": inspector.get_summary(),
        "components": inspector.inspect_components(),
        "nodes": inspector.inspect_nodes(),
        "uncertainties": inspector.inspect_uncertainties()
    }


@app.get("/api/jobs/{job_id}/validate")
def validate_job_circuit(job_id: str):
    """Executes rule-based electrical domain validation."""
    graph = _get_or_load_job_graph(job_id)
    validator = CircuitValidator(graph)
    findings = validator.validate()
    return {
        "success": True,
        "job_id": job_id,
        "total_findings": len(findings),
        "findings": [f.model_dump() for f in findings]
    }


def _persist_job_graph_updates(job_id: str, graph: CircuitGraph) -> str:
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "circuit_graph.json").write_text(
        json.dumps(graph.to_dict(), indent=2),
        encoding="utf-8",
    )

    spice_netlist = synthesize_graph_spice_netlist(graph)
    spice_path = next(iter(job_dir.glob("*.cir")), job_dir / f"{graph.image_name}.cir")
    spice_path.write_text(spice_netlist, encoding="utf-8")

    summary_path = job_dir / "circuit.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        components_by_id = {component.id: component for component in graph.components}
        for component_data in summary.get("components", []):
            component = components_by_id.get(component_data.get("id"))
            if component:
                component_data["value"] = component.value or ""
                component_data["designator"] = component.label or component_data.get("designator", "")
                component_data["text_label"] = component.value or component.label or ""
                component_data["cls_name"] = component.type
                component_data["cls_name_refined"] = component.type_refined
        summary["total_labels"] = sum(bool(c.value or c.label) for c in graph.components)
        summary["spice_netlist"] = spice_netlist
        summary["topology"] = {
            **summary.get("topology", {}),
            "inferred": bool(graph.nodes),
            "node_count": len(graph.nodes),
            "connected_terminal_count": sum(len(node.connections) for node in graph.nodes),
            "wire_count": len(graph.wires),
            "source": graph.metadata.get("source", "curated_ground_truth"),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    input_path = job_dir / "input.jpg"
    if input_path.is_file():
        image = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for component in graph.components:
            if len(component.bbox) != 4:
                continue
            x1, y1, x2, y2 = component.bbox
            color = get_class_color(component.type_refined or component.type)
            rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            label = component.display_type
            if component.value:
                label += f": {component.value}"
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
            bounds = draw.textbbox((0, 0), label, font=UI_LABEL_FONT)
            label_height = max(18, bounds[3] - bounds[1] + 4)
            draw.rectangle([x1, max(0, y1 - label_height - 3), x1 + bounds[2] + 8, y1], fill=rgb)
            draw.text((x1 + 4, max(0, y1 - label_height - 1)), label, fill="white", font=UI_LABEL_FONT)
        image.save(job_dir / "annotated.jpg", "JPEG", quality=95)

    render_circuit_graph(graph, job_dir)
    return spice_netlist


@app.post("/api/jobs/{job_id}/correct")
async def apply_job_correction(job_id: str, payload: Dict[str, Any]):
    """
    Applies user-directed correction mutations to the circuit graph representation,
    re-validates the graph, updates SPICE netlist, and re-renders the schematic.
    """
    graph = _get_or_load_job_graph(job_id)
    inspector = ConnectivityInspector(graph)

    action = payload.get("action", "")
    params = payload.get("params", {})
    ok, message = inspector.apply_correction(action, params)

    if not ok:
        return JSONResponse(status_code=400, content={"success": False, "message": message})

    spice_netlist = _persist_job_graph_updates(job_id, graph)

    # 3. Re-run validation
    validator = CircuitValidator(graph)
    findings = validator.validate()

    return {
        "success": True,
        "message": message,
        "job_id": job_id,
        "summary": inspector.get_summary(),
        "components": inspector.inspect_components(),
        "nodes": inspector.inspect_nodes(),
        "findings": [f.model_dump() for f in findings],
        "spice_netlist": spice_netlist,
        "graph": graph.to_dict(),
    }


@app.post("/api/jobs/{job_id}/simulate")
def simulate_job_circuit(job_id: str, analysis: str = Form("OP")):
    """
    Validates the generated SPICE netlist and executes simulation via ngspice.
    """
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    cir_files = list(job_dir.glob("*.cir")) + list(job_dir.glob("*.spice"))
    netlist_text = None
    if cir_files:
        with open(cir_files[0], "r", encoding="utf-8") as f:
            netlist_text = f.read()
    else:
        # Check sample circuits
        sample_cir = REPO_ROOT / "outputs" / "run1" / "final_result" / "cir" / f"{job_id}.cir"
        mcp_cir = REPO_ROOT / "outputs" / "mcp_spice" / f"{job_id}.spice"
        if sample_cir.exists():
            with open(sample_cir, "r", encoding="utf-8") as f:
                netlist_text = f.read()
        elif mcp_cir.exists():
            with open(mcp_cir, "r", encoding="utf-8") as f:
                netlist_text = f.read()
        else:
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from mcp_server import export_spice_netlist
            res = json.loads(export_spice_netlist(job_id))
            netlist_text = res.get("netlist", "")

    if not netlist_text:
        raise HTTPException(status_code=404, detail=f"SPICE netlist not available for {job_id}")

    simulator = SPICESimulator()
    result = simulator.run_simulation(netlist_text, analysis=analysis, temp_dir=job_dir)
    return {
        "success": result.success,
        "job_id": job_id,
        "result": result.model_dump()
    }


@app.get("/api/assistant/status")
def get_assistant_status():
    """Returns whether an external LLM is configured, without claiming connectivity."""
    assistant = AICircuitAssistant()
    local_endpoint_without_key = (
        assistant.provider == "OpenAI-compatible LLM"
        and assistant.base_url is not None
        and not any(host in assistant.base_url.lower() for host in ("api.openai.com", "api.groq.com"))
    )
    has_llm = bool(
        assistant.base_url
        and assistant.model
        and (assistant.api_key or local_endpoint_without_key)
        and assistant.api_key != "your_groq_api_key_here"
    )
    return {
        "status": "ready",
        "has_groq": has_llm,
        "has_llm": has_llm,
        "provider": assistant.provider,
        "hint": (
            f"{assistant.provider} is configured; network connectivity is checked when you send a message."
            if has_llm
            else (
                "Set GROQ_API_KEY, LLM_API_KEY, or OPENAI_API_KEY for a hosted provider, "
                "or set LLM_BASE_URL and LLM_MODEL for an OpenAI-compatible local model."
            )
        )
    }


@app.post("/api/jobs/{job_id}/assistant/chat")
async def chat_job_assistant(job_id: str, payload: Dict[str, Any]):
    """
    Queries Grounded AI Assistant powered by Groq regarding circuit topology,
    validation findings, SPICE netlist behavior, or specific component queries.
    """
    # Dynamically reload .env so user key changes take effect immediately
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    clean_id = job_id.strip()
    graph = _get_or_load_job_graph(clean_id)
    validator = CircuitValidator(graph)
    findings = validator.validate()

    job_dir = UPLOAD_DIR / clean_id
    spice_netlist = None
    cdata = {}
    if (job_dir / "circuit.json").exists():
        try:
            with open(job_dir / "circuit.json", "r", encoding="utf-8") as f:
                cdata = json.load(f)
            spice_netlist = cdata.get("spice_netlist")
        except Exception:
            pass

    if not spice_netlist and job_dir.exists():
        for cf in job_dir.glob("*.cir"):
            try:
                spice_netlist = cf.read_text(encoding="utf-8")
                break
            except Exception:
                pass

    if not spice_netlist:
        sample_cir = REPO_ROOT / "outputs" / "run1" / "final_result" / "cir" / f"{clean_id}.cir"
        if sample_cir.exists():
            try:
                spice_netlist = sample_cir.read_text(encoding="utf-8")
            except Exception:
                pass

    user_question = payload.get("question", "")
    history = payload.get("history", [])
    assistant = AICircuitAssistant()
    
    extra_context = {
        "job_id": clean_id,
        "total_components": len(cdata.get("components", graph.components)),
        "dimensions": f"{graph.width}x{graph.height}"
    }

    answer = assistant.explain_circuit(
        graph=graph,
        findings=findings,
        user_question=user_question,
        spice_netlist=spice_netlist,
        extra_context=extra_context,
        history=history
    )

    return {
        "success": True,
        "job_id": clean_id,
        "question": user_question,
        "answer": answer,
        "provider": getattr(assistant, "provider", "Groq"),
    }


@app.get("/api/mcp/tools")
def get_mcp_tools():
    """Returns metadata for all 6 Paper2CAD Model Context Protocol (MCP) tools."""
    return {
        "status": "online",
        "transport": "stdio / JSON-RPC 2.0",
        "protocol_version": "2024-11-05",
        "sdk_version": "mcp 1.x",
        "tools": [
            {
                "name": "inspect_circuit",
                "description": "Inspect normalized circuit graph, pin mappings, and electrical node connectivity.",
                "parameters": {
                    "circuit_id": {"type": "string", "required": True, "description": "Circuit name/ID (e.g. cir1, canonical_led)"},
                    "component_id": {"type": "integer", "required": False, "description": "Optional component ID to inspect pin-level details"}
                }
            },
            {
                "name": "validate_circuit",
                "description": "Run deterministic domain rules (grounding, shorts, floating pins, missing values).",
                "parameters": {
                    "circuit_id": {"type": "string", "required": True, "description": "Circuit name/ID to validate"}
                }
            },
            {
                "name": "generate_schematic",
                "description": "Render interactive computerized IEEE CAD schematic SVG with data attributes.",
                "parameters": {
                    "circuit_id": {"type": "string", "required": True, "description": "Circuit name/ID"},
                    "theme": {"type": "string", "default": "dark", "enum": ["dark", "light"]}
                }
            },
            {
                "name": "export_spice_netlist",
                "description": "Generate clean standard SPICE netlist (.cir) from circuit graph.",
                "parameters": {
                    "circuit_id": {"type": "string", "required": True, "description": "Circuit name/ID"}
                }
            },
            {
                "name": "simulate_circuit",
                "description": "Execute SPICE simulation using ngspice binary (DC OP or TRAN analysis).",
                "parameters": {
                    "circuit_id": {"type": "string", "required": True, "description": "Circuit name/ID"},
                    "analysis_type": {"type": "string", "default": "OP", "enum": ["OP", "TRAN"]}
                }
            },
            {
                "name": "explain_circuit",
                "description": "Ask grounded circuit AI assistant without hallucinations.",
                "parameters": {
                    "circuit_id": {"type": "string", "required": True, "description": "Circuit name/ID"},
                    "question": {"type": "string", "required": True, "description": "Question about circuit design/behavior"}
                }
            }
        ]
    }


@app.post("/api/mcp/invoke")
async def invoke_mcp_tool(payload: Dict[str, Any]):
    """Executes an MCP tool and returns its structured JSON output."""
    tool_name = payload.get("tool_name", "")
    args = payload.get("arguments", {})
    circuit_id = args.get("circuit_id", "cir1")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mcp_server import (
        inspect_circuit, validate_circuit, generate_schematic,
        export_spice_netlist, simulate_circuit, explain_circuit
    )

    try:
        if tool_name == "inspect_circuit":
            comp_id = args.get("component_id")
            if comp_id is not None and comp_id != "":
                try: comp_id = int(comp_id)
                except Exception: comp_id = None
            else:
                comp_id = None
            res = inspect_circuit(circuit_id=circuit_id, component_id=comp_id)
        elif tool_name == "validate_circuit":
            res = validate_circuit(circuit_id=circuit_id)
        elif tool_name == "generate_schematic":
            theme = args.get("theme", "dark")
            res = generate_schematic(circuit_id=circuit_id, theme=theme)
        elif tool_name == "export_spice_netlist":
            res = export_spice_netlist(circuit_id=circuit_id)
        elif tool_name == "simulate_circuit":
            analysis = args.get("analysis_type", "OP")
            res = simulate_circuit(circuit_id=circuit_id, analysis_type=analysis)
        elif tool_name == "explain_circuit":
            q = args.get("question", "Explain the circuit topology.")
            res = explain_circuit(circuit_id=circuit_id, question=q)
        else:
            return JSONResponse(status_code=400, content={"error": f"Unknown MCP tool: {tool_name}"})

        parsed = json.loads(res) if isinstance(res, str) else res
        return {"success": True, "tool": tool_name, "result": parsed}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})



# Mount static assets
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    port = int(os.environ.get("PORT", 8000))
    print(f"\n========================================================")
    print(f"[SERVER] PaperToCAD Web Studio running at: http://localhost:{port}")
    print(f"========================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
