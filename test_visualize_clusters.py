#!/usr/bin/env python3
"""
test_visualize_clusters.py
============================
Offline tests for visualize_clusters.py. Checks the actual selection/truncation/graph-construction
logic against a small synthetic clusters.json.

Run with:
    python -m unittest test_visualize_clusters.py -v
"""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from visualize_clusters import (
    address_type,
    build_degree_map,
    build_network,
    build_size_distribution_chart,
    cluster_color,
    load_data,
    select_clusters,
    truncate_large_cluster,
)


def make_synthetic_data() -> dict:
    """Three clusters of very different sizes, plus a merge_log wired up
    consistently with them, mimicking real clusters.json output."""
    big_addrs = [f"BIG{i}" for i in range(30)]
    med_addrs = [f"MED{i}" for i in range(5)]
    small_addrs = ["S0", "S1"]

    merge_log = []
    # chain the big cluster together (BIG0-BIG1, BIG1-BIG2, ...)
    for i in range(len(big_addrs) - 1):
        merge_log.append({"address_a": big_addrs[i], "address_b": big_addrs[i + 1],
                          "reason": f"H1:tx_big_{i}"})
    for i in range(len(med_addrs) - 1):
        merge_log.append({"address_a": med_addrs[i], "address_b": med_addrs[i + 1],
                          "reason": f"H2:tx_med_{i}"})
    merge_log.append({"address_a": "S0", "address_b": "S1", "reason": "H1:tx_small_0"})

    clusters = [
        {"cluster_id": 0, "size": len(big_addrs), "addresses": big_addrs},
        {"cluster_id": 1, "size": len(med_addrs), "addresses": med_addrs},
        {"cluster_id": 2, "size": len(small_addrs), "addresses": small_addrs},
    ]

    return {
        "seed_addresses": ["BIG0"],
        "stats": {"addresses_visited": 37},
        "clusters": clusters,
        "merge_log": merge_log,
    }


class TestAddressType(unittest.TestCase):
    def test_bech32(self):
        self.assertEqual(address_type("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"), "P2WPKH")

    def test_legacy(self):
        self.assertEqual(address_type("1BoatSLRHtKNngkdXEeobR76b53LETtpyT"), "P2PKH")


class TestLoadData(unittest.TestCase):
    def test_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            load_data("/nonexistent/path/clusters.json")

    def test_invalid_json_exits(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            path = f.name
        with self.assertRaises(SystemExit):
            load_data(path)

    def test_wrong_schema_exits(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"foo": "bar"}, f)
            path = f.name
        with self.assertRaises(SystemExit):
            load_data(path)

    def test_valid_file_loads(self):
        data = make_synthetic_data()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        loaded = load_data(path)
        self.assertEqual(len(loaded["clusters"]), 3)


class TestDegreeMap(unittest.TestCase):
    def test_counts_both_endpoints(self):
        merge_log = [
            {"address_a": "A", "address_b": "B", "reason": "H1:t1"},
            {"address_a": "A", "address_b": "C", "reason": "H1:t2"},
        ]
        deg = build_degree_map(merge_log)
        self.assertEqual(deg["A"], 2)
        self.assertEqual(deg["B"], 1)
        self.assertEqual(deg["C"], 1)


class TestClusterColor(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(cluster_color(3), cluster_color(3))

    def test_wraps_around_palette(self):
        # should never crash regardless of how large the cluster_id is
        c = cluster_color(999999)
        self.assertTrue(c.startswith("#"))


class TestSelectClusters(unittest.TestCase):
    def setUp(self):
        self.data = make_synthetic_data()

    def test_top_clusters_returns_largest_first(self):
        selected = select_clusters(self.data, top_clusters=2, focus_address=None, max_total_nodes=999)
        self.assertEqual([c["cluster_id"] for c in selected], [0, 1])

    def test_focus_address_returns_only_its_cluster(self):
        selected = select_clusters(self.data, top_clusters=None, focus_address="MED2", max_total_nodes=999)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["cluster_id"], 1)

    def test_focus_address_not_found_exits(self):
        with self.assertRaises(SystemExit):
            select_clusters(self.data, top_clusters=None, focus_address="NOT_THERE", max_total_nodes=999)

    def test_default_budget_always_includes_at_least_one_cluster(self):
        # budget smaller than even the smallest cluster -> must still return something
        selected = select_clusters(self.data, top_clusters=None, focus_address=None, max_total_nodes=1)
        self.assertGreaterEqual(len(selected), 1)

    def test_default_budget_stops_once_exceeded(self):
        # big cluster alone is 30; med is 5 more (35 total); small would push to 37
        selected = select_clusters(self.data, top_clusters=None, focus_address=None, max_total_nodes=36)
        ids = [c["cluster_id"] for c in selected]
        self.assertIn(0, ids)
        self.assertIn(1, ids)
        self.assertNotIn(2, ids)


class TestTruncateLargeCluster(unittest.TestCase):
    def test_no_truncation_when_under_limit(self):
        cluster = {"cluster_id": 0, "size": 5, "addresses": ["A", "B", "C", "D", "E"]}
        addrs, truncated = truncate_large_cluster(cluster, Counter(), max_nodes_per_cluster=10)
        self.assertFalse(truncated)
        self.assertEqual(len(addrs), 5)

    def test_truncates_and_keeps_highest_degree(self):
        cluster = {"cluster_id": 0, "size": 5, "addresses": ["A", "B", "C", "D", "E"]}
        degree = Counter({"A": 1, "B": 5, "C": 3, "D": 0, "E": 2})
        addrs, truncated = truncate_large_cluster(cluster, degree, max_nodes_per_cluster=2)
        self.assertTrue(truncated)
        self.assertEqual(addrs, ["B", "C"])  # the two highest-degree addresses


class TestBuildNetwork(unittest.TestCase):
    def setUp(self):
        self.data = make_synthetic_data()

    def test_all_selected_addresses_become_nodes(self):
        selected = select_clusters(self.data, top_clusters=2, focus_address=None, max_total_nodes=999)
        net, info = build_network(self.data, selected, max_nodes_per_cluster=999, physics=False)
        node_ids = {n["id"] for n in net.nodes}
        expected = set(selected[0]["addresses"]) | set(selected[1]["addresses"])
        self.assertEqual(node_ids, expected)
        self.assertEqual(info["node_count"], len(expected))

    def test_edges_excluded_when_endpoint_not_selected(self):
        # only the "small" cluster selected -> big/med merge_log edges must not appear
        selected = select_clusters(self.data, top_clusters=None, focus_address="S0", max_total_nodes=999)
        net, info = build_network(self.data, selected, max_nodes_per_cluster=999, physics=False)
        node_ids = {n["id"] for n in net.nodes}
        self.assertEqual(node_ids, {"S0", "S1"})
        for edge in net.edges:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)
        self.assertEqual(info["edge_count"], 1)

    def test_truncation_reduces_node_count_and_is_reported(self):
        selected = select_clusters(self.data, top_clusters=1, focus_address=None, max_total_nodes=999)  # the big one, size 30
        net, info = build_network(self.data, selected, max_nodes_per_cluster=10, physics=False)
        self.assertEqual(info["node_count"], 10)
        self.assertEqual(len(info["truncated"]), 1)
        cid, shown, total = info["truncated"][0]
        self.assertEqual((cid, shown, total), (0, 10, 30))

    def test_edge_style_reflects_heuristic(self):
        selected = select_clusters(self.data, top_clusters=None, focus_address="MED0", max_total_nodes=999)
        net, info = build_network(self.data, selected, max_nodes_per_cluster=999, physics=False)
        # all med-cluster edges are H2 in the synthetic data -> should render dashed
        self.assertTrue(all(e["dashes"] for e in net.edges))

    def test_no_nodes_when_no_clusters_selected(self):
        net, info = build_network(self.data, [], max_nodes_per_cluster=999, physics=False)
        self.assertEqual(info["node_count"], 0)
        self.assertEqual(info["edge_count"], 0)
        self.assertEqual(net.nodes, [])


class TestSizeDistributionChart(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        data = make_synthetic_data()
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "chart.png"
            build_size_distribution_chart(data, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_handles_empty_clusters_without_crashing(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "chart.png"
            build_size_distribution_chart({"clusters": []}, out_path)
            self.assertFalse(out_path.exists())  # nothing to plot -> function returns early


if __name__ == "__main__":
    unittest.main(verbosity=2)
