# Bitcoin Address Clustering

A self-contained, tested Python tool for clustering Bitcoin addresses that
likely belong to the same entity, using the two standard heuristics from
the academic literature. No local Bitcoin node required. It pulls live
data from a public block-explorer API as it crawls.

## Files

| File | Purpose |
|---|---|
| `bitcoin_address_clustering.py` | Clustering tool. To be run first. |
| `test_bitcoin_address_clustering.py` | Offline test suite (no network). |
| `visualize_clusters.py` | Turns `clusters.json` into interactive graph + chart. |
| `test_visualize_clusters.py` | Offline test suite for visualizer. |

## Quick start

```bash
pip install requests --break-system-packages   # or `pip install requests` in a venv

# Sanity check: run the offline test suite (takes < 1 second, no network)
python -m unittest test_bitcoin_address_clustering.py -v

# Cluster starting from a real address
python bitcoin_address_clustering.py 1BoatSLRHtKNngkdXEeobR76b53LETtpyT \
    --max-addresses 200 -v
```

This will crawl outward from the seed address, print progress as it goes,
and write full results to `clusters.json`.

## What it does

1. **Common-Input-Ownership Heuristic (H1)** — all input addresses spent
   in one transaction are assumed to be controlled by the same entity
   (a valid transaction requires all its inputs to be jointly authorized).
2. **Change-Address Heuristic (H2)** — in a simple 2-output payment
   transaction, the change output is identified using two independent
   signals (script-type match with the inputs, and "if address has
   never been used before"). The tool only accepts a confident,
   non-contradictory answer — ties abstain rather than guess.
3. **CoinJoin exclusion** — transactions with many equally-valued outputs
   (signature of Wasabi/Whirlpool/JoinMarket-style mixing) are
   detected and *excluded* from H1, since applying it there would merge
   unrelated participants' addresses together. (correctness safeguard)
4. Clustering is maintained with a **Union-Find (disjoint-set)** structure
   with path compression and union-by-rank (standard efficient
   approach, and the same idea used by BlockSci/GraphSense internally.)

Every merge is logged with *why* it happened (which heuristic, which
txid), so it's possible to audit any result in `clusters.json` under
`"merge_log"`.

## Visualizing results

```bash
pip install pyvis matplotlib --break-system-packages

python -m unittest test_visualize_clusters.py -v   # confirm it works (23 tests, offline)

python visualize_clusters.py clusters.json
```

This produces two files in `viz/`:

- **`cluster_graph.html`** — an interactive, force-directed network graph.
  Nodes are addresses (colored by cluster), edges are heuristic merges:
  solid blue = H1 (common-input), dashed orange = H2 (change-address).
  Hover any node or edge for details (address type, txid, etc). 
- **`cluster_size_distribution.png`** — a static chart showing how big
  clusters are relative to each other. Real crawls almost always
  produce one or two big clusters and a long tail of small
  ones. The chart is a better tool for seeing the overall shape, while the graph is
  better for inspecting one cluster closely.

Useful visualization flags:

```
--top-clusters N            visualize only the N largest clusters
--focus-address ADDR         show only the cluster containing this address
--max-nodes-per-cluster N    cap any one cluster to its N most-connected
                              addresses (default 150) — keeps huge clusters
                              (e.g. touching an exchange) from becoming an
                              unreadable; the graph tells when this happened 
                              and by how much
--max-total-nodes N          overall node budget when --top-clusters isn't
                              given (default 250)
--no-physics                  static layout instead of force-directed
                              (faster for big graphs)
```

Example: after a big crawl, zoom in on just the cluster containing
original seed address:

```bash
python visualize_clusters.py clusters.json --focus-address 1BoatSLRHtKNngkdXEeobR76b53LETtpyT
```

## Useful flags

```
--max-addresses N          stop after visiting N distinct addresses (default 300)
--max-tx-per-address N     cap transactions fetched per address (default 50)
--max-inputs-for-h1 N      skip H1 on transactions with more than N inputs —
                            guards against "super-clusters" from batched
                            exchange withdrawals (default 50)
--no-change-heuristic      disable H2 entirely
--no-reuse-check           disable the extra API call H2 makes per candidate
                            (faster, slightly less accurate)
--requests-per-second F    throttle API calls (default 3.0)
--base-url URL             point at a different Esplora-compatible API,
                            e.g. https://mempool.space/api as a fallback
                            if blockstream.info rate-limits 
--testnet                  use testnet instead of mainnet
--addresses-file FILE      seed addresses from a text file, one per line
-v / --verbose              debug-level logging
```

## Known limitations (please read)

- **Heuristics are probabilistic, not proof.** H1 can false-positive on
  batched exchange payouts (many customers' withdrawals in one tx look
  like "one entity"); H2 can miss change on non-standard wallets.
- **H2 only looks at simple 2-output transactions.** More complex shapes
  are intentionally skipped rather than guessed at.
- **Large services will still create big clusters** even with the
  `--max-inputs-for-h1` guard, because that's a real, correct consequence
  of the heuristic, not a bug — an exchange's hot wallet genuinely does
  co-sign many inputs in one transaction.
- **This is a research/educational tool**, not a forensic-grade product
  like BlockSci or GraphSense - can be used to *explore* and learn how
  clustering heuristics behave. Sanity-check any result before
  relying on it for anything that matters.
