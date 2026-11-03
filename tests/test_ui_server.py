import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.pipeline.circuit_data_model import CircuitGraph, ComponentModel, NodeConnection, NodeModel, TerminalModel
from ui import server
from ui.server import UI_LABEL_FONT, remove_colored_grid


class TestUIServer(unittest.TestCase):
    def test_assistant_status_does_not_expose_configured_model_string(self):
        configured_assistant = type(
            "ConfiguredAssistant",
            (),
            {
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
                "model": "private-model-config",
                "provider": "OpenAI",
            },
        )()
        with patch.object(server, "AICircuitAssistant", return_value=configured_assistant):
            status = server.get_assistant_status()

        self.assertTrue(status["has_llm"])
        self.assertNotIn("model", status)
        self.assertNotIn("private-model-config", str(status))

    def test_assistant_status_marks_hosted_provider_without_key_unconfigured(self):
        configured_assistant = type(
            "ConfiguredAssistant",
            (),
            {
                "api_key": None,
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "provider": "OpenAI-compatible LLM",
            },
        )()
        with patch.object(server, "AICircuitAssistant", return_value=configured_assistant):
            status = server.get_assistant_status()

        self.assertFalse(status["has_llm"])

    def test_annotation_font_supports_electrical_units(self):
        image = Image.new("RGB", (240, 40), "black")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), "Capacitor: C1 (100 µF, 10 Ω)", font=UI_LABEL_FONT, fill="white")

    def test_colored_grid_is_removed_without_erasing_dark_ink(self):
        image = Image.new("RGB", (160, 120), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, image.width, 6):
            draw.line((x, 0, x, image.height), fill="#e07070")
        for y in range(0, image.height, 6):
            draw.line((0, y, image.width, y), fill="#e07070")
        draw.line((83, 35, 83, 90), fill="black", width=2)

        cleaned, grid_removed = remove_colored_grid(image)

        self.assertTrue(grid_removed)
        self.assertGreater(min(cleaned.getpixel((30, 30))), 200)
        self.assertLess(max(cleaned.getpixel((83, 50))), 80)

    def test_sparse_colored_marks_do_not_trigger_grid_removal(self):
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).point((50, 50), fill="#e07070")

        cleaned, grid_removed = remove_colored_grid(image)

        self.assertFalse(grid_removed)
        self.assertEqual(cleaned.getpixel((50, 50)), (224, 112, 112))

    def test_component_value_updates_refresh_graph_summary_and_spice(self):
        graph = CircuitGraph(
            image_name="edited",
            width=100,
            height=80,
            components=[
                ComponentModel(
                    id=0,
                    type="resistor",
                    label="R1",
                    value="10k",
                    bbox=[10, 10, 30, 30],
                    terminals=[
                        TerminalModel(id=0, name="T1", x=10, y=20, component_id=0, connected_node_id=0),
                        TerminalModel(id=1, name="T2", x=30, y=20, component_id=0, connected_node_id=1),
                    ],
                )
            ],
            nodes=[
                NodeModel(id=0, name="N0", connections=[NodeConnection(component_id=0, terminal_id=0, terminal_name="T1")]),
                NodeModel(id=1, name="N1", connections=[NodeConnection(component_id=0, terminal_id=1, terminal_name="T2")]),
            ],
        )
        graph.components[0].value = "22k"

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(server, "UPLOAD_DIR", Path(temp_dir)), patch.object(server, "render_circuit_graph") as render:
                job_dir = Path(temp_dir) / "job"
                job_dir.mkdir()
                (job_dir / "circuit.json").write_text(
                    json.dumps({"components": [{"id": 0, "value": "10k"}]}),
                    encoding="utf-8",
                )

                spice = server._persist_job_graph_updates("job", graph)

                saved_graph = json.loads((job_dir / "circuit_graph.json").read_text(encoding="utf-8"))
                summary = json.loads((job_dir / "circuit.json").read_text(encoding="utf-8"))
                self.assertEqual(saved_graph["components"][0]["value"], "22k")
                self.assertEqual(summary["components"][0]["value"], "22k")
                self.assertIn("value=22k", spice)
                self.assertEqual((job_dir / "edited.cir").read_text(encoding="utf-8"), spice)
                render.assert_called_once_with(graph, job_dir)


if __name__ == "__main__":
    unittest.main()
