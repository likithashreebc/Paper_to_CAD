# -*- coding: utf-8 -*-
"""
Tests for Paper2CAD MCP Server Tools (Phase 7).
Directly tests invocations of all 6 exposed tools:
- inspect_circuit
- validate_circuit
- generate_schematic
- export_spice_netlist
- simulate_circuit
- explain_circuit
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline"))

from src.mcp_server import (
    _build_job_config,
    _missing_optional_pipeline_weights,
    inspect_circuit,
    process_sketch,
    validate_circuit,
    generate_schematic,
    export_spice_netlist,
    simulate_circuit,
    explain_circuit
)


class TestMCPServer(unittest.TestCase):

    def test_inspect_circuit_tool(self):
        res_str = inspect_circuit("cir1")
        res = json.loads(res_str)
        self.assertNotIn("error", res)
        self.assertEqual(res["circuit_id"], "cir1")
        self.assertIn("components", res)
        self.assertIn("nodes", res)
        self.assertGreater(len(res["components"]), 0)

    def test_circuit_id_rejects_path_segments(self):
        res = json.loads(inspect_circuit("../cir1"))
        self.assertIn("error", res)

    def test_process_sketch_rejects_missing_image(self):
        res = json.loads(process_sketch("missing-sketch.png"))
        self.assertIn("error", res)

    def test_job_config_isolates_pipeline_outputs(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            job_dir = root / "job"
            config = _build_job_config(
                {"paths": {"src_img": "shared-input", "final_result_root": "shared-output"}},
                input_dir,
                job_dir,
            )

        self.assertEqual(config["paths"]["src_img"], str(input_dir))
        self.assertEqual(config["paths"]["final_result_root"], str(job_dir / "final"))
        self.assertEqual(config["paths"]["spice_netlists"], str(job_dir / "spice"))

    def test_missing_optional_weights_identifies_skipped_stages(self):
        missing = _missing_optional_pipeline_weights({
            "weights": {
                "bjt_ckpt": "missing_bjt.pt",
                "fet_ckpt": "missing_fet.pt",
                "dcsrc_ckpt": "missing_dcsrc.pt",
                "pose3k_wts": "missing_pose3k.pth",
                "pose2k_wts": "missing_pose2k.pth",
            }
        })
        self.assertIn("DINOv2 subtype refinement", missing)
        self.assertIn("ViTPose port estimation", missing)
        self.assertEqual(len(missing["DINOv2 subtype refinement"]), 3)
        self.assertEqual(len(missing["ViTPose port estimation"]), 2)

    def test_validate_circuit_tool(self):
        res_str = validate_circuit("cir1")
        res = json.loads(res_str)
        self.assertNotIn("error", res)
        self.assertEqual(res["circuit_id"], "cir1")
        self.assertIn("findings", res)

    def test_generate_schematic_tool(self):
        res_str = generate_schematic("cir1", theme="dark")
        res = json.loads(res_str)
        self.assertNotIn("error", res)
        self.assertEqual(res["status"], "success")
        self.assertIn("files", res)
        self.assertIn("cad_dark_svg", res["files"])

    def test_export_spice_netlist_tool(self):
        res_str = export_spice_netlist("cir1")
        res = json.loads(res_str)
        self.assertNotIn("error", res)
        self.assertIn("netlist", res)
        self.assertIn(".END", res["netlist"].upper())

    def test_simulate_circuit_tool(self):
        res_str = simulate_circuit("cir1", analysis="OP")
        res = json.loads(res_str)
        self.assertNotIn("error", res)
        self.assertIn("available", res)
        self.assertIn("success", res)
        # Verify honest status report (whether ngspice is present or absent)
        self.assertIsInstance(res["available"], bool)

    def test_explain_circuit_tool(self):
        res_str = explain_circuit("cir1", question="What is connected to component 0?")
        res = json.loads(res_str)
        self.assertNotIn("error", res)
        self.assertEqual(res["circuit_id"], "cir1")
        self.assertIn("Component 0", res["explanation"])
        self.assertIn("Node", res["explanation"])


if __name__ == "__main__":
    unittest.main()
