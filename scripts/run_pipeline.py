#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform pipeline runner for Hand-Drawn Circuit Recognition.
Supports Windows, Linux, and macOS without requiring bash.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PIPELINE_DIR = SRC_DIR / "pipeline"

# Inject paths into sys.path and environment
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "HAWP"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "parseq-main"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "mmpose"))

from config_utils import get_default_config_path, load_paths_config


# Ensure stdout/stderr handle UTF-8 cleanly on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def cfg_get(cfg: dict, dotted_key: str, default=None):
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def run_cmd(cmd: list[str], desc: str, env=None):
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    start = time.time()
    res = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    duration = time.time() - start
    if res.returncode != 0:
        print(f"[FAIL] Failed ({desc}) with exit code {res.returncode} in {duration:.1f}s")
        sys.exit(res.returncode)
    print(f"[OK] Completed ({desc}) in {duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run Hand-Drawn Circuit Diagram Recognition Pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: configs/paths.yaml)")
    parser.add_argument("--stage", choices=["all", "yolo", "parseq", "suppress", "hawp", "merge", "wires", "connect", "final", "spice"],
                        default="all", help="Pipeline stage to run")
    parser.add_argument("--device", default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO detection confidence threshold")
    parser.add_argument("--skip-type-refinement", action="store_true",
                        help="Skip optional DINOv2 component subtype refinement")
    parser.add_argument("--skip-port-estimation", action="store_true",
                        help="Skip optional ViTPose component port estimation")
    args = parser.parse_args()

    cfg_file = args.config if args.config else str(get_default_config_path())
    print(f"Loading configuration from: {cfg_file}")
    cfg = load_paths_config(cfg_file)

    # Prepare environment variables
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    py_path = [
        str(PIPELINE_DIR),
        str(REPO_ROOT / "third_party" / "HAWP"),
        str(REPO_ROOT / "third_party" / "parseq-main"),
        str(REPO_ROOT / "third_party" / "mmpose"),
    ]
    if "PYTHONPATH" in env:
        py_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_path)
    env["HCD_PATHS_CONFIG"] = cfg_file

    # Extract configured paths
    src_img = cfg_get(cfg, "paths.src_img")
    yolo_model = cfg_get(cfg, "weights.yolo_best")
    hawp_cfg = cfg_get(cfg, "weights.hawp_cfg")
    hawp_ckpt = cfg_get(cfg, "weights.hawp_ckpt")
    yolo_exp = cfg_get(cfg, "paths.yolo_detect_exp")
    suppressed_img = cfg_get(cfg, "paths.suppressed_img")
    hawp_json = cfg_get(cfg, "paths.hawp_json")
    spice_dir = cfg_get(cfg, "paths.spice_netlists")

    yolo_project = str(Path(yolo_exp).parent)
    yolo_name = str(Path(yolo_exp).name)
    hawp_out = str(Path(hawp_json).parent)

    # Ensure output directories exist
    for d in [yolo_project, hawp_out, spice_dir]:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    py = sys.executable

    stage = args.stage

    # 1. YOLO
    if stage in ("all", "yolo"):
        print(f"\n{'='*60}")
        print(">>> [1/9] YOLOv10 Component Detection")
        print(f"{'='*60}")
        start_t = time.time()
        from ultralytics import YOLO
        yolo = YOLO(str(yolo_model))
        kwargs = {
            "source": str(src_img),
            "conf": args.conf,
            "iou": 0.40,
            "save_txt": True,
            "save_conf": True,
            "save_crop": True,
            "project": str(yolo_project),
            "name": str(yolo_name),
            "exist_ok": True,
        }
        if args.device:
            kwargs["device"] = args.device
        yolo.predict(**kwargs)
        print(f"[OK] YOLOv10 Prediction complete in {time.time() - start_t:.1f}s")
        run_cmd([py, str(PIPELINE_DIR / "scanify_folder.py")], "[1.1/9] Scanify Binarization", env=env)
        run_cmd([py, str(PIPELINE_DIR / "make_yolov10_crops_by_image.py")], "[1.2/9] Component Cropping", env=env)

    # 2. PARSeq
    if stage in ("all", "parseq"):
        run_cmd([py, str(PIPELINE_DIR / "fuse_yolo_parseq.py")], "[2/9] PARSeq OCR & Text Fusion", env=env)

    # 3. Suppress
    if stage in ("all", "suppress"):
        run_cmd([py, str(PIPELINE_DIR / "suppress_component_regions.py")], "[3/9] Component Region Suppression", env=env)

    # 4. HAWP
    if stage in ("all", "hawp"):
        hawp_cmd = [
            py, "-m", "hawp.fsl.predict_circuit_batch_fixed",
            "--config", str(hawp_cfg),
            "--ckpt", str(hawp_ckpt),
            "--input-dir", str(suppressed_img),
            "--output-dir", str(hawp_out),
            "--junc-th", "0.25"
        ]
        run_cmd(hawp_cmd, "[4/9] HAWP Wire & Junction Detection", env=env)

    # 5. Merge Points
    if stage in ("all", "merge"):
        run_cmd([py, str(PIPELINE_DIR / "merge_points.py")], "[5/9] Merge Junction Points", env=env)

    # 6. Wire enhancement / remove components
    if stage in ("all", "wires"):
        run_cmd([py, str(PIPELINE_DIR / "remove_components.py")], "[6/9] Wire Trace Cleaning & Enhancement", env=env)

    # 7. Connectivity & Ports
    if stage in ("all", "connect"):
        run_cmd([py, str(PIPELINE_DIR / "build_connections.py")], "[7.1/9] Connectivity Inference", env=env)
        if args.skip_type_refinement:
            print("[WARN] Skipping optional DINOv2 subtype refinement.")
        else:
            run_cmd([py, str(PIPELINE_DIR / "refine_component_types.py")], "[7.2/9] DINOv2 Subtype Refinement", env=env)
        if args.skip_port_estimation:
            print("[WARN] Skipping optional ViTPose port landmark estimation.")
        else:
            run_cmd([py, str(PIPELINE_DIR / "infer_port_vitpose.py")], "[7.3/9] ViTPose Port Landmark Estimation", env=env)

    # 8. Final JSON & Visuals
    if stage in ("all", "final"):
        run_cmd([py, str(PIPELINE_DIR / "build_final_json.py")], "[8/9] Build Final JSON & Visualizations", env=env)

    # 9. SPICE Netlists
    if stage in ("all", "spice"):
        spice_cmd = [
            py, str(PIPELINE_DIR / "build_spice_netlists.py"),
            "--cfg", cfg_file,
            "--output-dir", str(spice_dir)
        ]
        run_cmd(spice_cmd, "[9/9] Export SPICE Netlists", env=env)

    print("\n[DONE] Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
