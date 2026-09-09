#!/usr/bin/env python3
"""
visualize_clusters.py
======================

Turns the clusters.json produced by bitcoin_address_clustering.py into two
visual outputs:

  1. cluster_graph.html — an interactive, force-directed network graph
     (nodes = addresses, edges = heuristic merges) you can pan, zoom, drag,
     and hover over in any browser.  Best way to actually
     *see* why addresses got grouped together: edge color/style tells you
     which heuristic fired, and hovering an edge shows the exact txid.

  2. cluster_size_distribution.png — a static chart showing how cluster
     sizes are distributed. On real crawls this is almost always a sharp
     power-law (a handful of huge clusters — usually the seed's own wallet
     plus anything it touched that also touches an exchange — and a long
     tail of small ones). This chart is the better tool for seeing the *shape* of
     your results at a glance, while the graph is the better tool for
     inspecting a specific cluster closely.

The clustering output *is* graph-structured data 
(addresses as nodes, heuristic merges as edges).

Usage:
    python visualize_clusters.py clusters.json
    python visualize_clusters.py clusters.json --top-clusters 10
    python visualize_clusters.py clusters.json --focus-address bc1q...
    python visualize_clusters.py clusters.json --max-nodes-per-cluster 80 --no-physics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from pyvis.network import Network
except ImportError:
    sys.exit("This script requires pyvis. Install it with:\n"
              "    pip install pyvis --break-system-packages")

try:
    import matplotlib
    matplotlib.use("Agg")  # headless, no display needed
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("This script requires matplotlib. Install it with:\n"
              "    pip install matplotlib --break-system-packages")


# A ~20-color palette (tab20-derived) with good contrast on a dark background,
# cycled by cluster_id so adjacent clusters rarely look identical.
PALETTE = [
    "#5aa9e6", "#e6a15a", "#6ae68c", "#e65a8c", "#c15ae6",
    "#e6d95a", "#5ae6d9", "#e67e5a", "#8c5ae6", "#5ae67e",
    "#e65a5a", "#5a8ce6", "#a1e65a", "#e65ac1", "#5ae6a1",
    "#e6935a", "#7e5ae6", "#d9e65a", "#5ae65a", "#e65a93",
]

HEURISTIC_STYLE = {
    "H1": {"color": "#5aa9e6", "dashes": False, "label": "H1 (common-input)"},
    "H2": {"color": "#e6a15a", "dashes": True, "label": "H2 (change-address)"},
}


def address_type(addr: str) -> str:
    if addr.startswith(("bc1p", "tb1p")):
        return "P2TR"
    if addr.startswith(("bc1q", "tb1q", "bcrt1q")):
        return "P2WPKH" if len(addr) == 42 else "P2WSH"
    if addr.startswith(("3", "2")):
        return "P2SH"
    if addr.startswith(("1", "m", "n")):
        return "P2PKH"
    return "UNKNOWN"


def cluster_color(cluster_id: int) -> str:
    return PALETTE[cluster_id % len(PALETTE)]


def load_data(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")
    try:
        with open(p) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON: {e}")
    if "clusters" not in data or "merge_log" not in data:
        sys.exit(f"{path} doesn't look like a clusters.json produced by "
                  "bitcoin_address_clustering.py (missing 'clusters' or 'merge_log').")
    return data


def build_degree_map(merge_log: list[dict]) -> Counter:
    """Count how many heuristic merges each address participated in.
    Used both as a node-size signal and to pick which addresses to keep
    when a cluster is too large to render in full."""
    deg: Counter = Counter()
    for e in merge_log:
        deg[e["address_a"]] += 1
        deg[e["address_b"]] += 1
    return deg


def select_clusters(data: dict, top_clusters: Optional[int],
                     focus_address: Optional[str], max_total_nodes: int) -> list[dict]:
    clusters = data["clusters"]  # already sorted largest-first by the main tool

    if focus_address:
        addr_to_cluster = {a: c for c in clusters for a in c["addresses"]}
        c = addr_to_cluster.get(focus_address)
        if c is None:
            sys.exit(f"Address {focus_address!r} isn't in this clusters.json "
                      f"(it wasn't visited during the crawl that produced it).")
        return [c]

    if top_clusters is not None:
        return clusters[:top_clusters]

    # Default: greedily take the largest clusters until we'd exceed the
    # node budget, but always include at least one so a single huge crawl
    # still produces *something* to look at.
    selected: list[dict] = []
    total = 0
    for c in clusters:
        if selected and total + c["size"] > max_total_nodes:
            break
        selected.append(c)
        total += c["size"]
    return selected


def truncate_large_cluster(cluster: dict, degree: Counter,
                            max_nodes_per_cluster: int) -> tuple[list[str], bool]:
    """If a cluster is bigger than max_nodes_per_cluster, keep only its
    most-connected (highest-degree) addresses — these are the ones most
    likely to actually explain the cluster's structure, rather than an
    arbitrary or alphabetical subset."""
    addrs = cluster["addresses"]
    if len(addrs) <= max_nodes_per_cluster:
        return addrs, False
    ranked = sorted(addrs, key=lambda a: degree.get(a, 0), reverse=True)
    return ranked[:max_nodes_per_cluster], True


def build_network(data: dict, selected_clusters: list[dict],
                   max_nodes_per_cluster: int, physics: bool) -> tuple[Network, dict]:
    """Constructs the pyvis Network object (kept separate from writing the
    HTML so the resulting node/edge sets can be unit-tested directly)."""
    net = Network(height="850px", width="100%", bgcolor="#111418",
                  font_color="#e6e6e6", directed=False, notebook=False,
                  cdn_resources="in_line")
    if physics:
        net.barnes_hut(gravity=-2500, central_gravity=0.15, spring_length=110,
                        spring_strength=0.02, damping=0.85)
    else:
        net.toggle_physics(False)

    degree = build_degree_map(data["merge_log"])
    node_set: set[str] = set()
    info = {"truncated": [], "node_count": 0, "edge_count": 0}

    for c in selected_clusters:
        addrs, was_truncated = truncate_large_cluster(c, degree, max_nodes_per_cluster)
        if was_truncated:
            info["truncated"].append((c["cluster_id"], len(addrs), c["size"]))
        color = cluster_color(c["cluster_id"])
        for a in addrs:
            node_set.add(a)
            deg = degree.get(a, 0)
            size = 10 + min(30, deg * 3)
            short = a if len(a) <= 14 else f"{a[:6]}…{a[-4:]}"
            title = (f"{a}\ncluster #{c['cluster_id']} ({c['size']} addresses)\n"
                     f"type: {address_type(a)}\nheuristic links: {deg}")
            net.add_node(a, label=short, title=title, color=color, size=size)

    for e in data["merge_log"]:
        a, b, reason = e["address_a"], e["address_b"], e["reason"]
        if a not in node_set or b not in node_set:
            continue
        heuristic = reason.split(":", 1)[0]
        txid = reason.split(":", 1)[1] if ":" in reason else ""
        style = HEURISTIC_STYLE.get(heuristic, {"color": "#999999", "dashes": False,
                                                  "label": heuristic})
        net.add_edge(a, b, color=style["color"], dashes=style["dashes"],
                    title=f"{style['label']} · tx {txid}" if txid else style["label"])
        info["edge_count"] += 1

    info["node_count"] = len(node_set)
    net.set_options("""
    var options = {
      "interaction": {"hover": true, "tooltipDelay": 120, "navigationButtons": true,
                        "keyboard": true},
      "physics": {"stabilization": {"iterations": 200}}
    }
    """)
    return net, info


def build_size_distribution_chart(data: dict, out_path: Path, top_n: int = 30) -> None:
    clusters = data["clusters"]
    sizes = sorted((c["size"] for c in clusters), reverse=True)
    if not sizes:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    top = sizes[:top_n]
    axes[0].bar(range(1, len(top) + 1), top, color="#5aa9e6")
    axes[0].set_xlabel("Cluster rank (by size)")
    axes[0].set_ylabel("Number of addresses")
    axes[0].set_title(f"Top {len(top)} clusters by size")

    ranks = range(1, len(sizes) + 1)
    axes[1].loglog(ranks, sizes, marker=".", linestyle="none", color="#e6a15a")
    axes[1].set_xlabel("Cluster rank (log)")
    axes[1].set_ylabel("Cluster size (log)")
    axes[1].set_title("Full cluster-size distribution (log-log)")

    fig.suptitle(f"{len(sizes)} clusters total, {sum(sizes)} addresses")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize clusters.json as an interactive network graph "
                    "plus a cluster-size distribution chart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("clusters_json", help="path to clusters.json from bitcoin_address_clustering.py")
    p.add_argument("--out-dir", default="viz", help="directory to write outputs into")
    p.add_argument("--top-clusters", type=int, default=None,
                    help="visualize only the N largest clusters (overrides --max-total-nodes)")
    p.add_argument("--max-total-nodes", type=int, default=250,
                    help="if --top-clusters isn't given, include as many top clusters as "
                        "fit under this many total nodes")
    p.add_argument("--max-nodes-per-cluster", type=int, default=150,
                    help="cap any single cluster to its N most-connected addresses "
                        "(prevents one huge cluster turning the graph into a hairball)")
    p.add_argument("--focus-address", default=None,
                    help="show only the cluster containing this one address")
    p.add_argument("--no-physics", action="store_true",
                    help="disable force-directed physics simulation (faster for big graphs, "
                        "static layout)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data = load_data(args.clusters_json)

    if not data["clusters"]:
        sys.exit("No clusters found in this file — nothing to visualize.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_clusters(data, args.top_clusters, args.focus_address, args.max_total_nodes)
    if not selected:
        sys.exit("Nothing selected to visualize — check --focus-address / --top-clusters.")

    net, info = build_network(data, selected, args.max_nodes_per_cluster,
                              physics=not args.no_physics)
    html_path = out_dir / "cluster_graph.html"
    net.write_html(str(html_path), notebook=False)

    chart_path = out_dir / "cluster_size_distribution.png"
    build_size_distribution_chart(data, chart_path)

    print(f"Visualized {len(selected)} cluster(s): {info['node_count']} nodes, "
          f"{info['edge_count']} edges.")
    for cid, shown, total in info["truncated"]:
        print(f"  cluster #{cid}: showing {shown}/{total} addresses "
              f"(kept the most-connected ones — raise --max-nodes-per-cluster to see more)")
    print(f"\nInteractive graph : {html_path}  (open in any browser, no internet needed)")
    print(f"Size distribution : {chart_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
