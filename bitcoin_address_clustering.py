#!/usr/bin/env python3
"""
bitcoin_address_clustering.py
==============================

A self-contained Bitcoin address clustering tool. Given one or more seed
addresses, it crawls outward through their transaction history (using a
public block-explorer API) and groups addresses
that are likely controlled by the same real-world entity, using two
well-established heuristics from the blockchain-analysis literature:

  H1. Common-Input-Ownership Heuristic
      All input addresses spent within a single transaction are assumed
      to be controlled by the same entity, since a valid transaction
      requires all inputs to be signed by their owner(s) in concert.
      Reference: Meiklejohn et al., "A Fistful of Bitcoins" (2013);
                 already noted informally in Satoshi's whitepaper.

  H2. Change-Address Heuristic
      In a typical 2-output "payment" transaction, one output pays the
      recipient and the other returns leftover value ("change") to the
      sender. We identify the likely change output using two supporting
      signals and require agreement (or an unambiguous single signal):
        (a) Script-type match: wallets usually generate change in their
            own default address format, so if all inputs share one
            address type and only one output matches it, that output is
            probably change.
        (b) One-time-address / no-reuse: change addresses are freshly
            generated and, at the time of the transaction, should not
            have been seen anywhere on the chain before (tx_count == 1).
      Reference: Androulaki et al. (2013); Meiklejohn et al. (2013).

Important correctness safeguard — CoinJoin exclusion:
  H1 is invalid on CoinJoin-style transactions (Wasabi, Whirlpool,
  JoinMarket, ...), which are explicitly constructed to have many
  *independent* inputs and equally-valued outputs. Applying H1 blindly
  to such transactions is the single most common source of bad, over-
  merged clusters in naive implementations. This tool detects the
  telltale "many equal-valued outputs" signature and skips both
  heuristics for those transactions.

Clustering itself is done with a Union-Find (disjoint-set) structure
with path compression and union-by-rank, which is the standard,
efficient way to maintain merge-only groups over a large, growing set
of addresses.

Data source:
  Defaults to the public Blockstream Esplora API (blockstream.info),
  which needs no API key. Any Esplora-
  compatible instance works via --base-url (e.g. https://mempool.space/api
  as a fallback if blockstream.info rate-limits).

Usage:
    python bitcoin_address_clustering.py <address> [<address> ...] \\
        --max-addresses 300 --max-tx-per-address 50

    python bitcoin_address_clustering.py --addresses-file seeds.txt \\
        --out my_clusters.json -v

Output:
    A JSON file with cluster membership, per-merge audit log (which
    heuristic + which transaction caused each merge), and run stats.
    A human-readable summary is also printed to stdout.

Limitations (please read):
  - Heuristics are probabilistic, not proof. Both can produce false
    positives (e.g. batched exchange withdrawals look like H1 merges
    even though many distinct customers are involved) and false
    negatives (e.g. wallets that avoid change, or use PayJoin/CoinJoin).
  - The change heuristic only fires on simple 2-output transactions and
    intentionally declines to guess when signals conflict.
  - --max-inputs-for-h1 guards against "super-cluster" explosions from
    large batched transactions (common with exchanges); tune it down if
    seed addresses touch high-volume services.
  - This is a research/educational tool, not a forensic-grade product.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("This tool requires the 'requests' package. Install it with:\n"
              "    pip install requests --break-system-packages\n"
              "(drop --break-system-packages if you're in a virtualenv)")

logger = logging.getLogger("btc_cluster")

DEFAULT_BASE_URL = "https://blockstream.info/api"
TESTNET_BASE_URL = "https://blockstream.info/testnet/api"


# ---------------------------------------------------------------------------
# Union-Find (disjoint-set) over string address keys
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set structure with path compression + union by rank.

    Keyed on arbitrary hashable objects (Bitcoin addresses, as strings),
    rather than the classic array-of-ints version, since our universe of
    elements is discovered incrementally and isn't known up front.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        # Audit trail: every successful merge, with the reason it happened.
        self.edges: list[tuple[str, str, str]] = []

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression: relink every visited node directly to the root.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str, reason: str = "") -> bool:
        """Merge the sets containing a and b. Returns True if a merge
        actually happened (False if they were already in the same set)."""
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.edges.append((a, b, reason))
        return True

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> dict[str, list[str]]:
        """Return {root_address: [members...]} for every known address."""
        out: dict[str, list[str]] = {}
        for x in list(self.parent):
            r = self.find(x)
            out.setdefault(r, []).append(x)
        return out


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------

def address_type(addr: str) -> str:
    """Classify a Bitcoin address by its script type, based on prefix.
    Used for the script-type-match change-detection signal."""
    if addr.startswith(("bc1p", "tb1p")):
        return "P2TR"
    if addr.startswith(("bc1q", "tb1q", "bcrt1q")):
        return "P2WPKH" if len(addr) == 42 else "P2WSH"
    if addr.startswith(("3", "2")):
        return "P2SH"
    if addr.startswith(("1", "m", "n")):
        return "P2PKH"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Transaction model (source-agnostic — the same shape whichever API/backend
# fills it in, which keeps the heuristics testable without any network)
# ---------------------------------------------------------------------------

@dataclass
class TxIO:
    address: Optional[str]
    value: int  # satoshis


@dataclass
class Tx:
    txid: str
    inputs: list[TxIO]
    outputs: list[TxIO]
    is_coinbase: bool = False


def parse_esplora_tx(raw: dict) -> Tx:
    """Convert a raw Esplora API transaction JSON object into a Tx."""
    inputs: list[TxIO] = []
    is_coinbase = False
    for vin in raw.get("vin", []):
        if vin.get("is_coinbase"):
            is_coinbase = True
            continue
        prevout = vin.get("prevout") or {}
        inputs.append(TxIO(
            address=prevout.get("scriptpubkey_address"),
            value=prevout.get("value", 0) or 0,
        ))
    outputs: list[TxIO] = []
    for vout in raw.get("vout", []):
        outputs.append(TxIO(
            address=vout.get("scriptpubkey_address"),
            value=vout.get("value", 0) or 0,
        ))
    return Tx(txid=raw["txid"], inputs=inputs, outputs=outputs, is_coinbase=is_coinbase)


# ---------------------------------------------------------------------------
# CoinJoin-style detection (safety valve for Heuristic 1)
# ---------------------------------------------------------------------------

def looks_like_coinjoin(tx: Tx, min_participants: int = 3, equal_output_ratio: float = 0.5) -> bool:
    """Flag transactions that look like CoinJoin / mixing transactions,
    where the common-input-ownership heuristic does NOT hold by design
    (inputs come from multiple independent participants).

    Signature used: a large share of the outputs share one exact value
    (the standard "equal-output CoinJoin" shape used by Wasabi,
    Whirlpool, JoinMarket, etc.), together with multiple inputs.
    This is intentionally conservative (a few false negatives are far
    less damaging than false positives, since a missed CoinJoin just
    means we don't merge — a wrongly-applied H1 corrupts the cluster).
    """
    if len(tx.inputs) < 2 or len(tx.outputs) < min_participants:
        return False
    value_counts: dict[int, int] = {}
    for o in tx.outputs:
        value_counts[o.value] = value_counts.get(o.value, 0) + 1
    if not value_counts:
        return False
    _, dominant_count = max(value_counts.items(), key=lambda kv: kv[1])
    return (dominant_count >= min_participants
            and (dominant_count / len(tx.outputs)) >= equal_output_ratio)


# ---------------------------------------------------------------------------
# Chain data source abstraction (lets heuristics be tested without network)
# ---------------------------------------------------------------------------

class ChainDataSource:
    def get_address_transactions(self, address: str, max_txs: int) -> list[Tx]:
        raise NotImplementedError

    def get_address_tx_count(self, address: str) -> Optional[int]:
        """Total number of transactions (confirmed + mempool) an address
        has ever appeared in. Used for the address-reuse change signal."""
        raise NotImplementedError


class EsploraSource(ChainDataSource):
    """Live data source backed by a public Esplora-compatible REST API
    (default: blockstream.info; also works against mempool.space/api or
    a self-hosted esplora/electrs instance)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, requests_per_second: float = 3.0,
                 timeout: float = 15.0, max_retries: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "btc-address-clustering/1.0"})
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call = 0.0
        self.timeout = timeout
        self.max_retries = max_retries
        self._tx_count_cache: dict[str, Optional[int]] = {}

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def _get(self, path: str) -> Optional[requests.Response]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning("network error on %s (attempt %d/%d): %s", url, attempt, self.max_retries, e)
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return resp
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 60)
                logger.warning("rate limited on %s, backing off %.1fs", url, wait)
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                wait = min(2 ** attempt, 30)
                logger.warning("server error %d on %s, retrying in %.1fs", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            logger.warning("unexpected status %d on %s", resp.status_code, url)
            return resp
        raise RuntimeError(f"giving up on {url} after {self.max_retries} attempts")

    def get_address_transactions(self, address: str, max_txs: int) -> list[Tx]:
        txs: list[Tx] = []
        last_seen: Optional[str] = None
        while len(txs) < max_txs:
            path = f"/address/{address}/txs" if last_seen is None else f"/address/{address}/txs/chain/{last_seen}"
            resp = self._get(path)
            if resp is None or resp.status_code == 404:
                break
            try:
                batch = resp.json()
            except ValueError:
                logger.warning("non-JSON response for %s, stopping pagination", path)
                break
            if not batch:
                break
            for raw in batch:
                try:
                    txs.append(parse_esplora_tx(raw))
                except (KeyError, TypeError) as e:
                    logger.warning("skipping malformed tx from %s: %s", address, e)
                    continue
                if len(txs) >= max_txs:
                    break
            if len(batch) < 25:
                break  # short page == last page (esplora pages confirmed txs 25 at a time)
            last_seen = batch[-1]["txid"]
        return txs[:max_txs]

    def get_address_tx_count(self, address: str) -> Optional[int]:
        if address in self._tx_count_cache:
            return self._tx_count_cache[address]
        resp = self._get(f"/address/{address}")
        count: Optional[int] = None
        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json()
                cs = data.get("chain_stats", {})
                ms = data.get("mempool_stats", {})
                count = cs.get("tx_count", 0) + ms.get("tx_count", 0)
            except ValueError:
                count = None
        self._tx_count_cache[address] = count
        return count


# ---------------------------------------------------------------------------
# Heuristic 2: change-address detection
# ---------------------------------------------------------------------------

def detect_change_output(tx: Tx, source: ChainDataSource,
                          use_network_reuse_check: bool = True) -> Optional[str]:
    """Try to identify the change output of a simple 2-output transaction.

    Returns the change address, or None if the shape doesn't apply or the
    available signals don't give a confident, non-contradictory answer.
    Deliberately conservative: ties and missing signals both return None.
    """
    if len(tx.outputs) != 2 or not tx.inputs:
        return None

    out_a, out_b = tx.outputs
    if not out_a.address or not out_b.address or out_a.address == out_b.address:
        return None

    input_addrs = {i.address for i in tx.inputs if i.address}
    if not input_addrs:
        return None

    # An output paying back to one of the tx's own inputs isn't a useful
    # change candidate for our purposes (rare, but guard against it).
    candidates = [o for o in (out_a, out_b) if o.address not in input_addrs]
    if len(candidates) == 0:
        return None

    # Signal (a): script-type match with the input address type.
    input_types = {address_type(a) for a in input_addrs}
    type_signal: Optional[str] = None
    if len(input_types) == 1:
        dominant_type = next(iter(input_types))
        a_matches = address_type(out_a.address) == dominant_type
        b_matches = address_type(out_b.address) == dominant_type
        if a_matches and not b_matches:
            type_signal = out_a.address
        elif b_matches and not a_matches:
            type_signal = out_b.address

    # Signal (b): address has never been used before this transaction.
    reuse_signal: Optional[str] = None
    if use_network_reuse_check:
        a_count = source.get_address_tx_count(out_a.address)
        b_count = source.get_address_tx_count(out_b.address)
        a_fresh = a_count == 1
        b_fresh = b_count == 1
        if a_fresh and not b_fresh:
            reuse_signal = out_a.address
        elif b_fresh and not a_fresh:
            reuse_signal = out_b.address

    if type_signal and reuse_signal:
        return type_signal if type_signal == reuse_signal else None  # disagreement -> abstain
    return type_signal or reuse_signal


# ---------------------------------------------------------------------------
# Crawler / clustering engine
# ---------------------------------------------------------------------------

@dataclass
class CrawlConfig:
    max_addresses: int = 300
    max_tx_per_address: int = 50
    max_inputs_for_h1: int = 50       # skip H1 on very large (likely batched/exchange) txs
    coinjoin_min_participants: int = 3
    use_change_heuristic: bool = True
    use_network_reuse_check: bool = True
    max_queue_fanout_per_tx: int = 25  # bound how many new addresses one tx can enqueue


@dataclass
class CrawlStats:
    addresses_visited: int = 0
    tx_processed: int = 0
    coinjoin_skipped: int = 0
    h1_merges: int = 0
    h2_merges: int = 0
    api_errors: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def crawl_and_cluster(seeds: Iterable[str], source: ChainDataSource,
                       config: CrawlConfig) -> tuple[UnionFind, CrawlStats]:
    uf = UnionFind()
    visited: set[str] = set()
    queued: set[str] = set()
    queue: deque[str] = deque()
    seen_tx: set[str] = set()
    stats = CrawlStats()

    for s in seeds:
        uf.add(s)
        if s not in queued:
            queue.append(s)
            queued.add(s)

    try:
        while queue and len(visited) < config.max_addresses:
            addr = queue.popleft()
            queued.discard(addr)
            if addr in visited:
                continue
            visited.add(addr)
            uf.add(addr)

            try:
                txs = source.get_address_transactions(addr, config.max_tx_per_address)
            except Exception as e:  # network/backend failure for this address only
                logger.warning("failed to fetch transactions for %s: %s", addr, e)
                stats.api_errors += 1
                continue

            logger.info("[%d/%d] %s -> %d tx (h1=%d h2=%d clusters merges so far)",
                        len(visited), config.max_addresses, addr, len(txs),
                        stats.h1_merges, stats.h2_merges)

            for tx in txs:
                if tx.txid in seen_tx or tx.is_coinbase:
                    continue
                seen_tx.add(tx.txid)
                stats.tx_processed += 1

                if looks_like_coinjoin(tx, min_participants=config.coinjoin_min_participants):
                    stats.coinjoin_skipped += 1
                    continue

                input_addrs = [i.address for i in tx.inputs if i.address]

                # Heuristic 1: common-input-ownership.
                if 1 < len(input_addrs) <= config.max_inputs_for_h1:
                    anchor = input_addrs[0]
                    for other in input_addrs[1:]:
                        if uf.union(anchor, other, reason=f"H1:{tx.txid}"):
                            stats.h1_merges += 1

                # Heuristic 2: change-address detection.
                if config.use_change_heuristic and input_addrs:
                    change_addr = detect_change_output(
                        tx, source, use_network_reuse_check=config.use_network_reuse_check)
                    if change_addr and uf.union(input_addrs[0], change_addr, reason=f"H2:{tx.txid}"):
                        stats.h2_merges += 1

                # Discover new addresses to explore, regardless of whether a
                # heuristic fired on this tx (we still want to walk the graph).
                fanout = 0
                candidates = input_addrs + [o.address for o in tx.outputs if o.address]
                for a in candidates:
                    if fanout >= config.max_queue_fanout_per_tx:
                        break
                    if a in visited or a in queued:
                        continue
                    if len(visited) + len(queued) >= config.max_addresses:
                        break
                    queue.append(a)
                    queued.add(a)
                    fanout += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — returning partial results collected so far.")

    stats.addresses_visited = len(visited)
    return uf, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cluster Bitcoin addresses via common-input-ownership and "
                    "change-address heuristics, crawling live from a public "
                    "block-explorer API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("addresses", nargs="*", help="one or more seed Bitcoin addresses")
    p.add_argument("--addresses-file", help="text file with one address per line, "
                                            "added to any addresses given on the command line")
    p.add_argument("--max-addresses", type=int, default=300,
                    help="stop after visiting this many distinct addresses")
    p.add_argument("--max-tx-per-address", type=int, default=50,
                    help="max transactions fetched per address (pagination cap)")
    p.add_argument("--max-inputs-for-h1", type=int, default=50,
                    help="skip Heuristic 1 on transactions with more inputs than this "
                        "(guards against exchange/batch-payout super-clusters)")
    p.add_argument("--no-change-heuristic", action="store_true",
                    help="disable Heuristic 2 (change-address detection)")
    p.add_argument("--no-reuse-check", action="store_true",
                    help="disable the extra API call per change-candidate that checks "
                        "address reuse (faster, slightly less accurate H2)")
    p.add_argument("--requests-per-second", type=float, default=3.0,
                    help="rate limit for API calls")
    p.add_argument("--base-url", default=None,
                    help=f"Esplora-compatible API base URL "
                        f"(default: {DEFAULT_BASE_URL}; try https://mempool.space/api "
                        f"as a fallback)")
    p.add_argument("--testnet", action="store_true",
                    help=f"use the testnet API ({TESTNET_BASE_URL}) instead of mainnet")
    p.add_argument("--out", default="clusters.json", help="output JSON file")
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds: list[str] = list(args.addresses)
    if args.addresses_file:
        with open(args.addresses_file) as f:
            seeds.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    seeds = list(dict.fromkeys(seeds))  # de-dupe, preserve order
    if not seeds:
        build_arg_parser().error("no addresses given (positionally or via --addresses-file)")

    base_url = args.base_url or (TESTNET_BASE_URL if args.testnet else DEFAULT_BASE_URL)

    config = CrawlConfig(
        max_addresses=args.max_addresses,
        max_tx_per_address=args.max_tx_per_address,
        max_inputs_for_h1=args.max_inputs_for_h1,
        use_change_heuristic=not args.no_change_heuristic,
        use_network_reuse_check=not args.no_reuse_check,
    )
    source = EsploraSource(base_url=base_url, requests_per_second=args.requests_per_second)

    logger.info("Starting crawl from %d seed address(es) against %s", len(seeds), base_url)
    logger.info("Limits: max_addresses=%d max_tx_per_address=%d max_inputs_for_h1=%d",
                config.max_addresses, config.max_tx_per_address, config.max_inputs_for_h1)

    uf, stats = crawl_and_cluster(seeds, source, config)

    groups = uf.groups()
    clusters = sorted(groups.values(), key=len, reverse=True)

    result = {
        "seed_addresses": seeds,
        "stats": {**stats.as_dict(), "clusters_found": len(clusters)},
        "clusters": [
            {"cluster_id": i, "size": len(members), "addresses": sorted(members)}
            for i, members in enumerate(clusters)
        ],
        "merge_log": [
            {"address_a": a, "address_b": b, "reason": r} for a, b, r in uf.edges
        ],
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Done. Visited {stats.addresses_visited} addresses, found {len(clusters)} clusters.")
    print(f"Transactions processed: {stats.tx_processed}  |  CoinJoin-like skipped: {stats.coinjoin_skipped}")
    print(f"H1 merges: {stats.h1_merges}  |  H2 merges: {stats.h2_merges}  |  API errors: {stats.api_errors}")
    if clusters:
        print(f"Largest cluster size: {len(clusters[0])}")
    print()
    for seed in seeds:
        if seed not in uf.parent:
            print(f"  {seed} -> not visited (max-addresses limit reached first)")
            continue
        root = uf.find(seed)
        cluster = groups[root]
        print(f"  {seed} -> cluster of {len(cluster)} address(es)")
    print(f"\nFull results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
