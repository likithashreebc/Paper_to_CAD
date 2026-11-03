import unittest

from PIL import Image, ImageDraw

from ui.server import correct_diode_misclassified_as_battery
from src.pipeline.cad_schematic_renderer import CADSchematicRenderer
from src.pipeline.circuit_data_model import (
    CircuitGraph,
    ComponentModel,
    NodeConnection,
    NodeModel,
    TerminalModel,
    WireSegmentModel,
)


class TestGraphBasedSchematicRendering(unittest.TestCase):
    def _parallel_load_graph(self):
        components = [
            ComponentModel(
                id=0,
                type="voltage.ac",
                bbox=[20, 40, 60, 100],
                terminals=[
                    TerminalModel(id=0, name="T1", side="bottom", x=40, y=100, component_id=0, connected_node_id=0),
                    TerminalModel(id=1, name="T2", side="top", x=40, y=40, component_id=0, connected_node_id=1),
                ],
            ),
            ComponentModel(
                id=1,
                type="diode",
                bbox=[80, 30, 130, 50],
                terminals=[
                    TerminalModel(id=2, name="A", side="left", x=80, y=40, component_id=1, connected_node_id=1),
                    TerminalModel(id=3, name="K", side="right", x=130, y=40, component_id=1, connected_node_id=2),
                ],
            ),
            ComponentModel(
                id=2,
                type="capacitor.unpolarized",
                bbox=[145, 70, 165, 110],
                terminals=[
                    TerminalModel(id=4, name="T1", side="bottom", x=155, y=110, component_id=2, connected_node_id=0),
                    TerminalModel(id=5, name="T2", side="top", x=155, y=70, component_id=2, connected_node_id=2),
                ],
            ),
            ComponentModel(
                id=3,
                type="resistor",
                bbox=[185, 70, 205, 110],
                terminals=[
                    TerminalModel(id=6, name="T1", side="bottom", x=195, y=110, component_id=3, connected_node_id=0),
                    TerminalModel(id=7, name="T2", side="top", x=195, y=70, component_id=3, connected_node_id=2),
                ],
            ),
        ]
        nodes = [
            NodeModel(
                id=0,
                name="N0",
                connections=[
                    NodeConnection(component_id=0, terminal_id=0, terminal_name="T1"),
                    NodeConnection(component_id=2, terminal_id=4, terminal_name="T1"),
                    NodeConnection(component_id=3, terminal_id=6, terminal_name="T1"),
                ],
            ),
            NodeModel(
                id=1,
                name="N1",
                connections=[
                    NodeConnection(component_id=0, terminal_id=1, terminal_name="T2"),
                    NodeConnection(component_id=1, terminal_id=2, terminal_name="A"),
                ],
            ),
            NodeModel(
                id=2,
                name="N2",
                connections=[
                    NodeConnection(component_id=1, terminal_id=3, terminal_name="K"),
                    NodeConnection(component_id=2, terminal_id=5, terminal_name="T2"),
                    NodeConnection(component_id=3, terminal_id=7, terminal_name="T2"),
                ],
            ),
        ]
        wires = [
            WireSegmentModel(id=0, start_x=40, start_y=100, end_x=155, end_y=110, node_id=0),
            WireSegmentModel(id=1, start_x=155, start_y=110, end_x=195, end_y=110, node_id=0),
            WireSegmentModel(id=2, start_x=40, start_y=40, end_x=80, end_y=40, node_id=1),
            WireSegmentModel(id=3, start_x=130, start_y=40, end_x=155, end_y=70, node_id=2),
            WireSegmentModel(id=4, start_x=155, start_y=70, end_x=195, end_y=70, node_id=2),
        ]
        return CircuitGraph(
            image_name="parallel_load",
            width=220,
            height=140,
            components=components,
            nodes=nodes,
            wires=wires,
        )

    def test_graph_renderer_uses_detected_wires_and_component_types(self):
        graph = self._parallel_load_graph()
        graph.wires[0].confidence = 0.05

        svg = CADSchematicRenderer().render_connected_graph_svg(graph)

        self.assertIn('id="wires"', svg)
        self.assertIn('data-comp-id="1"', svg)
        self.assertIn("<polygon", svg)
        self.assertNotIn(">GND<", svg)
        self.assertIn(
            '<path d="M 295.0 449.3 L 713.9 449.3 L 713.9 485.7"/>',
            svg,
        )
        self.assertEqual(svg.count('stroke-width="3" stroke-linecap="round" stroke-linejoin="round"'), 1)

    def test_graph_renderer_labels_missing_topology(self):
        graph = self._parallel_load_graph()
        graph.nodes = []
        graph.wires = []

        svg = CADSchematicRenderer().render_connected_graph_svg(graph)

        self.assertIn("TOPOLOGY NOT INFERRED - COMPONENTS ONLY", svg)

    def test_graph_renderer_connects_multiterminal_component_pins(self):
        graph = self._parallel_load_graph()
        opamp = graph.components[0]
        opamp.type = "operational_amplifier"
        opamp.bbox = [50, 40, 130, 100]
        opamp.terminals = [
            TerminalModel(id=20, name="In-", side="left", x=50, y=55, component_id=0),
            TerminalModel(id=21, name="In+", side="left", x=50, y=85, component_id=0),
            TerminalModel(id=22, name="Out", side="right", x=130, y=70, component_id=0),
        ]

        svg = CADSchematicRenderer().render_connected_graph_svg(graph)

        self.assertEqual(svg.count('data-terminal-id="20"'), 1)
        self.assertEqual(svg.count('data-terminal-id="21"'), 1)
        self.assertEqual(svg.count('data-terminal-id="22"'), 1)

    def test_graph_renderer_creates_missing_wires_from_shared_net_terminals(self):
        graph = self._parallel_load_graph()
        graph.wires = []
        graph.nodes[0].connections = [
            NodeConnection(component_id=0, terminal_id=0, terminal_name="T1"),
            NodeConnection(component_id=2, terminal_id=4, terminal_name="T1"),
        ]
        graph.components[0].terminals[0].connected_node_id = 0
        graph.components[2].terminals[0].connected_node_id = 0

        svg = CADSchematicRenderer().render_connected_graph_svg(graph)

        self.assertIn('<path data-node-id="0"', svg)
        self.assertNotIn("TOPOLOGY NOT INFERRED - COMPONENTS ONLY", svg)

    def test_graph_renderer_displays_detected_component_values(self):
        graph = self._parallel_load_graph()
        graph.components[0].value = "9V"
        graph.components[1].value = "4.7k"

        svg = CADSchematicRenderer().render_connected_graph_svg(graph)

        self.assertIn(">9 V</text>", svg)
        self.assertIn(">4.7 k</text>", svg)
        self.assertIn("VALUE REQUIRED", svg)
        self.assertIn(".cad-val { font-family: monospace; font-size: 24px;", svg)

    def test_battery_prediction_is_corrected_for_diode_geometry(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((12, 10, 48, 40), fill="black", width=3)
        draw.line((12, 70, 48, 40), fill="black", width=3)
        draw.line((58, 12, 58, 68), fill="black", width=3)
        graph = self._parallel_load_graph()
        diode = graph.components[1]
        diode.type = "voltage.battery"
        diode.bbox = [0, 0, 80, 80]
        diode.terminals[0].side = "left"
        diode.terminals[1].side = "right"

        corrected = correct_diode_misclassified_as_battery(graph, image)

        self.assertTrue(corrected)
        self.assertEqual(diode.type, "diode")
        self.assertEqual([terminal.name for terminal in diode.terminals], ["A", "K"])

    def test_battery_shape_is_not_reclassified_as_diode(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((30, 10, 30, 70), fill="black", width=3)
        draw.line((50, 20, 50, 60), fill="black", width=3)
        graph = self._parallel_load_graph()
        battery = graph.components[1]
        battery.type = "voltage.battery"
        battery.bbox = [0, 0, 80, 80]

        corrected = correct_diode_misclassified_as_battery(graph, image)

        self.assertFalse(corrected)
        self.assertEqual(battery.type, "voltage.battery")


if __name__ == "__main__":
    unittest.main()
