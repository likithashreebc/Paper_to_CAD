import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline"))

import build_connections


class TestBuildConnections(unittest.TestCase):
    def setUp(self):
        self.original_near_k = build_connections.NEAR_K
        self.original_connect_radius = build_connections.CONNECT_RADIUS
        self.original_cross_cc_strict = build_connections.CROSS_CC_STRICT
        build_connections.NEAR_K = 10
        build_connections.CONNECT_RADIUS = 1
        build_connections.CROSS_CC_STRICT = True

    def tearDown(self):
        build_connections.NEAR_K = self.original_near_k
        build_connections.CONNECT_RADIUS = self.original_connect_radius
        build_connections.CROSS_CC_STRICT = self.original_cross_cc_strict

    def test_missing_led_terminal_is_inferred_from_component_bbox(self):
        points = [
            {
                "id": 4,
                "type": "component_terminal",
                "x": 472.0,
                "y": 302.0,
                "matches": [{
                    "comp_id": 5,
                    "cls_id": 20,
                    "cls_name": "diode.light_emitting",
                    "side": "bottom",
                    "edge_dist": 0.4,
                }],
            },
        ]
        components = [{
            "id": 5,
            "cls_id": 20,
            "cls_name": "diode.light_emitting",
            "bbox": [456.0, 266.0, 510.0, 302.0],
        }]

        endpoints, _, _ = build_connections.build_endpoints(points, components)
        diode_terminals = [endpoint for endpoint in endpoints if endpoint.get("comp_id") == 5]

        self.assertEqual(len(diode_terminals), 2)
        self.assertEqual(
            {(endpoint["side"], endpoint["x"], endpoint["y"]) for endpoint in diode_terminals},
            {("bottom", 472.0, 302.0), ("top", 472.0, 266.0)},
        )
        self.assertTrue(next(endpoint for endpoint in diode_terminals if endpoint["side"] == "top")["inferred_terminal"])

    def test_inferred_terminal_only_connects_to_a_nearby_endpoint(self):
        wire_img = np.zeros((100, 100), dtype=np.uint8)
        endpoints = [
            {
                "eid": 0,
                "kind": "terminal",
                "comp_id": 1,
                "x": 50.0,
                "y": 50.0,
                "cc_label": 0,
                "inferred_terminal": True,
                "inferred_connect_radius": 12.0,
            },
            {"eid": 1, "kind": "terminal", "comp_id": 2, "x": 58.0, "y": 50.0, "cc_label": 0},
            {"eid": 2, "kind": "terminal", "comp_id": 3, "x": 85.0, "y": 50.0, "cc_label": 0},
        ]

        edges = build_connections.connect_by_nearest(
            endpoints, wire_img, max_search_dist=100, base_ratio=0.82
        )

        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["eid1"], edges[0]["eid2"]), (0, 1))
        self.assertEqual(edges[0]["dir"], "inferred")

    def test_strong_wire_path_connects_different_component_labels(self):
        wire_img = np.zeros((100, 200), dtype=np.uint8)
        wire_img[50, 10:96] = 255
        wire_img[50, 101:191] = 255
        endpoints = [
            {"eid": 0, "kind": "terminal", "x": 10.0, "y": 50.0, "cc_label": 1},
            {"eid": 1, "kind": "node", "x": 190.0, "y": 50.0, "cc_label": 2},
        ]

        edges = build_connections.connect_by_nearest(
            endpoints, wire_img, max_search_dist=250, base_ratio=0.82
        )

        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["eid1"], edges[0]["eid2"]), (0, 1))
        self.assertEqual(edges[0]["dir"], "nearest")

    def test_different_component_labels_without_wire_support_stay_disconnected(self):
        wire_img = np.zeros((100, 200), dtype=np.uint8)
        wire_img[50, 10:40] = 255
        wire_img[50, 160:191] = 255
        endpoints = [
            {"eid": 0, "kind": "terminal", "x": 10.0, "y": 50.0, "cc_label": 1},
            {"eid": 1, "kind": "node", "x": 190.0, "y": 50.0, "cc_label": 2},
        ]

        edges = build_connections.connect_by_nearest(
            endpoints, wire_img, max_search_dist=250, base_ratio=0.82
        )

        self.assertEqual(edges, [])

    def test_wire_cannot_short_across_both_pins_of_a_component(self):
        wire_img = np.zeros((100, 100), dtype=np.uint8)
        wire_img[10:91, 50] = 255
        endpoints = [
            {"eid": 0, "kind": "node", "x": 50.0, "y": 10.0, "cc_label": 1},
            {"eid": 1, "kind": "node", "x": 50.0, "y": 90.0, "cc_label": 2},
            {"eid": 2, "kind": "terminal", "comp_id": 7, "x": 50.0, "y": 40.0, "cc_label": 3},
            {"eid": 3, "kind": "terminal", "comp_id": 7, "x": 50.0, "y": 60.0, "cc_label": 4},
        ]

        edges = build_connections.connect_by_nearest(
            endpoints, wire_img, max_search_dist=100, base_ratio=0.82
        )
        edge_pairs = {
            tuple(sorted((edge["eid1"], edge["eid2"])))
            for edge in edges
        }

        self.assertNotIn((0, 1), edge_pairs)
        self.assertNotIn((2, 3), edge_pairs)

    def test_long_wire_path_connects_endpoints_beyond_default_search_radius(self):
        wire_img = np.zeros((357, 560), dtype=np.uint8)
        for x in list(range(76, 266)) + list(range(276, 466)):
            y = round(50 - (x - 76) * 8 / 389)
            wire_img[y, x] = 255
        endpoints = [
            {"eid": 0, "kind": "node", "x": 76.0, "y": 50.0, "cc_label": 1},
            {"eid": 1, "kind": "node", "x": 465.0, "y": 42.0, "cc_label": 2},
        ]
        max_search_dist = build_connections.compute_max_search_dist_from_image(wire_img)

        edges = build_connections.connect_by_nearest(
            endpoints, wire_img, max_search_dist=max_search_dist, base_ratio=0.82
        )

        self.assertGreater(max_search_dist, 389)
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["eid1"], edges[0]["eid2"]), (0, 1))

    def test_long_endpoints_without_wire_support_stay_disconnected(self):
        wire_img = np.zeros((357, 560), dtype=np.uint8)
        endpoints = [
            {"eid": 0, "kind": "node", "x": 76.0, "y": 50.0, "cc_label": 1},
            {"eid": 1, "kind": "node", "x": 465.0, "y": 42.0, "cc_label": 2},
        ]
        max_search_dist = build_connections.compute_max_search_dist_from_image(wire_img)

        edges = build_connections.connect_by_nearest(
            endpoints, wire_img, max_search_dist=max_search_dist, base_ratio=0.82
        )

        self.assertEqual(edges, [])


if __name__ == "__main__":
    unittest.main()
