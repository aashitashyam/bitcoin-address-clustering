#!/usr/bin/env python3
"""
test_bitcoin_address_clustering.py
===================================

Offline test suite. Exercises the Union-Find
structure, both clustering heuristics, CoinJoin detection, and a full
end-to-end crawl against a small synthetic (fake) blockchain, so the core
logic can be verified without depending on a live API.

Run with:
    python -m unittest test_bitcoin_address_clustering.py -v
"""

import unittest
from typing import Optional

from bitcoin_address_clustering import (
    UnionFind,
    address_type,
    Tx,
    TxIO,
    looks_like_coinjoin,
    detect_change_output,
    ChainDataSource,
    CrawlConfig,
    crawl_and_cluster,
    parse_esplora_tx,
)


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------

class TestUnionFind(unittest.TestCase):
    def test_singleton_is_its_own_root(self):
        uf = UnionFind()
        uf.add("A")
        self.assertEqual(uf.find("A"), "A")

    def test_union_merges_two_sets(self):
        uf = UnionFind()
        self.assertTrue(uf.union("A", "B", "test"))
        self.assertEqual(uf.find("A"), uf.find("B"))

    def test_union_returns_false_when_already_merged(self):
        uf = UnionFind()
        uf.union("A", "B", "r1")
        self.assertFalse(uf.union("A", "B", "r2"))
        # no duplicate edge recorded for a no-op union
        self.assertEqual(len(uf.edges), 1)

    def test_transitive_merging(self):
        uf = UnionFind()
        uf.union("A", "B", "r1")
        uf.union("B", "C", "r2")
        # A and C should be connected even though never unioned directly
        self.assertTrue(uf.connected("A", "C"))

    def test_groups_are_correct_and_exhaustive(self):
        uf = UnionFind()
        uf.union("A", "B", "r")
        uf.union("C", "D", "r")
        uf.add("E")  # isolated
        groups = uf.groups()
        sizes = sorted(len(v) for v in groups.values())
        self.assertEqual(sizes, [1, 2, 2])
        total_members = sum(len(v) for v in groups.values())
        self.assertEqual(total_members, 5)

    def test_find_auto_adds_unknown_element(self):
        uf = UnionFind()
        # find() on a never-seen element shouldn't crash (KeyError etc.)
        self.assertEqual(uf.find("never-seen"), "never-seen")

    def test_large_chain_does_not_blow_the_stack(self):
        uf = UnionFind()
        n = 5000
        for i in range(n - 1):
            uf.union(str(i), str(i + 1), "chain")
        self.assertTrue(uf.connected("0", str(n - 1)))


# ---------------------------------------------------------------------------
# address_type
# ---------------------------------------------------------------------------

class TestAddressType(unittest.TestCase):
    def test_legacy_p2pkh(self):
        self.assertEqual(address_type("1BoatSLRHtKNngkdXEeobR76b53LETtpyT"), "P2PKH")

    def test_p2sh(self):
        self.assertEqual(address_type("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"), "P2SH")

    def test_bech32_p2wpkh(self):
        self.assertEqual(address_type("bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"), "P2WPKH")

    def test_bech32_p2wsh_longer(self):
        addr = "bc1q" + "a" * 58  # 62 chars total -> P2WSH-shaped
        self.assertEqual(address_type(addr), "P2WSH")

    def test_taproot(self):
        self.assertEqual(
            address_type("bc1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr"),
            "P2TR",
        )

    def test_unknown_falls_back_gracefully(self):
        # deliberately avoids the 1/3/2/m/n/bc1/tb1 prefixes this function checks
        self.assertEqual(address_type("xyz_gibberish_not_a_real_address"), "UNKNOWN")


# ---------------------------------------------------------------------------
# CoinJoin detection
# ---------------------------------------------------------------------------

class TestCoinJoinDetection(unittest.TestCase):
    def test_flags_classic_equal_output_coinjoin(self):
        tx = Tx(
            txid="cj1",
            inputs=[TxIO(f"in{i}", 100_000) for i in range(5)],
            outputs=[TxIO(f"out{i}", 20_000) for i in range(5)],  # all equal
        )
        self.assertTrue(looks_like_coinjoin(tx))

    def test_does_not_flag_ordinary_payment(self):
        tx = Tx(
            txid="normal1",
            inputs=[TxIO("A", 50_000), TxIO("B", 30_000)],
            outputs=[TxIO("C", 60_000), TxIO("D", 19_000)],
        )
        self.assertFalse(looks_like_coinjoin(tx))

    def test_single_input_never_flagged(self):
        # can't be a CoinJoin with only one participant's input
        tx = Tx(
            txid="single_in",
            inputs=[TxIO("A", 100_000)],
            outputs=[TxIO(f"o{i}", 10_000) for i in range(5)],
        )
        self.assertFalse(looks_like_coinjoin(tx))

    def test_mixed_values_not_flagged(self):
        tx = Tx(
            txid="varied",
            inputs=[TxIO("A", 100_000), TxIO("B", 50_000), TxIO("C", 25_000)],
            outputs=[TxIO("D", 40_000), TxIO("E", 55_000), TxIO("F", 70_000)],
        )
        self.assertFalse(looks_like_coinjoin(tx))

    def test_partial_equal_outputs_below_ratio_not_flagged(self):
        # 3 outputs share a value out of 10 total outputs -> ratio 0.3, below default 0.5
        tx = Tx(
            txid="partial",
            inputs=[TxIO("A", 500_000), TxIO("B", 500_000)],
            outputs=[TxIO(f"o{i}", 10_000) for i in range(3)]
                    + [TxIO(f"p{i}", 1000 * i + 111) for i in range(7)],
        )
        self.assertFalse(looks_like_coinjoin(tx))


# ---------------------------------------------------------------------------
# Fake chain data source (for change-heuristic + full crawl tests)
# ---------------------------------------------------------------------------

class FakeSource(ChainDataSource):
    """In-memory stand-in for the live Esplora API."""

    def __init__(self, txs_by_address: dict[str, list[Tx]], tx_counts: dict[str, int]):
        self.txs_by_address = txs_by_address
        self.tx_counts = tx_counts
        self.reuse_check_calls = 0

    def get_address_transactions(self, address: str, max_txs: int) -> list[Tx]:
        return self.txs_by_address.get(address, [])[:max_txs]

    def get_address_tx_count(self, address: str) -> Optional[int]:
        self.reuse_check_calls += 1
        return self.tx_counts.get(address)


# ---------------------------------------------------------------------------
# detect_change_output
# ---------------------------------------------------------------------------

class TestChangeDetection(unittest.TestCase):
    def _make_source(self, tx_counts):
        return FakeSource(txs_by_address={}, tx_counts=tx_counts)

    def test_ignores_non_two_output_tx(self):
        tx = Tx("t1", inputs=[TxIO("A", 1000)],
                outputs=[TxIO("B", 500), TxIO("C", 400), TxIO("D", 90)])
        self.assertIsNone(detect_change_output(tx, self._make_source({})))

    def test_ignores_tx_with_no_inputs(self):
        tx = Tx("t1", inputs=[], outputs=[TxIO("B", 500), TxIO("C", 400)])
        self.assertIsNone(detect_change_output(tx, self._make_source({})))

    def test_type_and_reuse_signals_agree(self):
        # inputs are bech32 P2WPKH; C matches type + is fresh -> confident change
        tx = Tx(
            "t1",
            inputs=[TxIO("bc1qsenderaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100_000)],
            outputs=[
                TxIO("1LegacyRecipientAddressxxxxxxxxxxx", 60_000),   # payment, different type, reused
                TxIO("bc1qchangeaddressbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 39_500),  # change: same type
            ],
        )
        source = self._make_source({
            "1LegacyRecipientAddressxxxxxxxxxxx": 7,   # reused many times -> not fresh
            "bc1qchangeaddressbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": 1,  # first appearance -> fresh
        })
        result = detect_change_output(tx, source, use_network_reuse_check=True)
        self.assertEqual(result, "bc1qchangeaddressbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

    def test_conflicting_signals_abstain(self):
        # type-match points at output A, but reuse-freshness points at output B
        tx = Tx(
            "t1",
            inputs=[TxIO("bc1qsenderaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100_000)],
            outputs=[
                TxIO("bc1qoutputaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60_000),  # type matches
                TxIO("1LegacyOutputBxxxxxxxxxxxxxxxxxxxx", 39_500),               # is "fresh"
            ],
        )
        source = self._make_source({
            "bc1qoutputaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": 9,   # reused -> not fresh
            "1LegacyOutputBxxxxxxxxxxxxxxxxxxxx": 1,                # fresh, but wrong type
        })
        result = detect_change_output(tx, source, use_network_reuse_check=True)
        self.assertIsNone(result)

    def test_no_signal_available_abstains(self):
        # both outputs same type as inputs, both "fresh" -> genuine tie, no basis to pick
        tx = Tx(
            "t1",
            inputs=[TxIO("bc1qsenderaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100_000)],
            outputs=[
                TxIO("bc1qoutaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 60_000),
                TxIO("bc1qoutbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 39_500),
            ],
        )
        source = self._make_source({
            "bc1qoutaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": 1,
            "bc1qoutbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": 1,
        })
        result = detect_change_output(tx, source, use_network_reuse_check=True)
        self.assertIsNone(result)

    def test_type_signal_alone_without_reuse_check(self):
        tx = Tx(
            "t1",
            inputs=[TxIO("bc1qsenderaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 100_000)],
            outputs=[
                TxIO("1LegacyRecipientxxxxxxxxxxxxxxxxxxx", 60_000),
                TxIO("bc1qchangebbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 39_500),
            ],
        )
        source = self._make_source({})  # unused when reuse check disabled
        result = detect_change_output(tx, source, use_network_reuse_check=False)
        self.assertEqual(result, "bc1qchangebbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        self.assertEqual(source.reuse_check_calls, 0)


# ---------------------------------------------------------------------------
# parse_esplora_tx
# ---------------------------------------------------------------------------

class TestParseEsploraTx(unittest.TestCase):
    def test_parses_typical_transaction(self):
        raw = {
            "txid": "abc123",
            "vin": [
                {"is_coinbase": False, "prevout": {"scriptpubkey_address": "A", "value": 1000}},
                {"is_coinbase": False, "prevout": {"scriptpubkey_address": "B", "value": 2000}},
            ],
            "vout": [
                {"scriptpubkey_address": "C", "value": 2500},
                {"scriptpubkey_address": "D", "value": 400},
            ],
        }
        tx = parse_esplora_tx(raw)
        self.assertEqual(tx.txid, "abc123")
        self.assertEqual([i.address for i in tx.inputs], ["A", "B"])
        self.assertEqual([o.value for o in tx.outputs], [2500, 400])
        self.assertFalse(tx.is_coinbase)

    def test_coinbase_input_is_flagged_and_excluded(self):
        raw = {
            "txid": "coinbase_tx",
            "vin": [{"is_coinbase": True}],
            "vout": [{"scriptpubkey_address": "MinerAddr", "value": 625_000_000}],
        }
        tx = parse_esplora_tx(raw)
        self.assertTrue(tx.is_coinbase)
        self.assertEqual(tx.inputs, [])

    def test_missing_scriptpubkey_address_becomes_none(self):
        # e.g. OP_RETURN outputs have no address
        raw = {
            "txid": "opreturn_tx",
            "vin": [{"is_coinbase": False, "prevout": {"scriptpubkey_address": "A", "value": 1000}}],
            "vout": [
                {"value": 0},  # OP_RETURN, no scriptpubkey_address key at all
                {"scriptpubkey_address": "B", "value": 900},
            ],
        }
        tx = parse_esplora_tx(raw)
        self.assertIsNone(tx.outputs[0].address)
        self.assertEqual(tx.outputs[1].address, "B")


# ---------------------------------------------------------------------------
# Full crawl_and_cluster integration test against a small synthetic graph
# ---------------------------------------------------------------------------

class TestCrawlAndCluster(unittest.TestCase):
    """
    Builds a tiny synthetic transaction graph by hand and checks that the
    crawler produces exactly the clusters we'd expect:

        tx1: inputs [A, B]        outputs [C(reused), D(fresh, type-match)]
             -> H1 merges A & B
             -> H2 merges A & D (D is the change output)

        tx2: inputs [D]           outputs [E(reused), F(fresh, type-match)]
             -> H2 merges D & F  (so A, B, D, F end up in one cluster)

        tx_cj: a CoinJoin-shaped transaction touching otherwise-unconnected
               addresses G..K -> must NOT trigger H1 (they must stay apart)

        Z is a completely unrelated, isolated address.
    """

    def setUp(self):
        # All addresses share the same (bech32) type except C and E, which
        # are legacy-type "payment recipients" so the type-match signal
        # cleanly points at D and F as change.
        self.tx1 = Tx(
            "tx1",
            inputs=[TxIO("A", 100_000), TxIO("B", 50_000)],
            outputs=[TxIO("1CReused", 120_000), TxIO("bc1qDfresh" + "x" * 32, 29_500)],
        )
        self.tx2 = Tx(
            "tx2",
            inputs=[TxIO("bc1qDfresh" + "x" * 32, 29_500)],
            outputs=[TxIO("1EReused", 20_000), TxIO("bc1qFfresh" + "x" * 32, 9_300)],
        )
        cj_inputs = [TxIO(f"G{i}", 100_000) for i in range(4)]
        cj_outputs = [TxIO(f"H{i}", 24_000) for i in range(4)]
        self.tx_cj = Tx("tx_cj", inputs=cj_inputs, outputs=cj_outputs)

        d_addr = "bc1qDfresh" + "x" * 32
        f_addr = "bc1qFfresh" + "x" * 32

        txs_by_address = {
            "A": [self.tx1],
            "B": [self.tx1],
            "1CReused": [self.tx1],
            d_addr: [self.tx1, self.tx2],
            "1EReused": [self.tx2],
            f_addr: [self.tx2],
        }
        for i in range(4):
            txs_by_address[f"G{i}"] = [self.tx_cj]
            txs_by_address[f"H{i}"] = [self.tx_cj]
        txs_by_address["Z"] = []  # isolated, no transactions at all

        tx_counts = {
            "1CReused": 9,       # reused -> not fresh
            d_addr: 1,           # this IS tx1's change, its first appearance
            "1EReused": 5,       # reused -> not fresh
            f_addr: 1,           # this IS tx2's change
        }

        self.source = FakeSource(txs_by_address, tx_counts)
        self.d_addr = d_addr
        self.f_addr = f_addr

    def test_change_and_common_input_merges_propagate_correctly(self):
        config = CrawlConfig(max_addresses=100, max_tx_per_address=50)
        uf, stats = crawl_and_cluster(["A"], self.source, config)

        # A, B (H1) and D, F (H2 chained through D) should all be one cluster.
        self.assertTrue(uf.connected("A", "B"))
        self.assertTrue(uf.connected("A", self.d_addr))
        self.assertTrue(uf.connected("A", self.f_addr))

        # The "reused" payment-recipient addresses must NOT be swept in.
        self.assertFalse(uf.connected("A", "1CReused"))
        self.assertFalse(uf.connected("A", "1EReused"))

        self.assertGreaterEqual(stats.h1_merges, 1)
        self.assertGreaterEqual(stats.h2_merges, 2)

    def test_coinjoin_participants_are_not_merged(self):
        config = CrawlConfig(max_addresses=100, max_tx_per_address=50)
        uf, stats = crawl_and_cluster(["G0"], self.source, config)
        # None of the 4 CoinJoin inputs should be merged with each other.
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(uf.connected(f"G{i}", f"G{j}"))
        self.assertGreaterEqual(stats.coinjoin_skipped, 1)

    def test_isolated_address_stays_alone(self):
        config = CrawlConfig(max_addresses=100, max_tx_per_address=50)
        uf, stats = crawl_and_cluster(["Z"], self.source, config)
        self.assertEqual(uf.groups()[uf.find("Z")], ["Z"])

    def test_max_addresses_limit_is_respected(self):
        config = CrawlConfig(max_addresses=2, max_tx_per_address=50)
        uf, stats = crawl_and_cluster(["A"], self.source, config)
        self.assertLessEqual(stats.addresses_visited, 2)

    def test_max_inputs_for_h1_guard_disables_merge_on_big_tx(self):
        # With the cap set below the CoinJoin tx's input count... but that tx
        # is already excluded by CoinJoin detection. Test the guard directly
        # with a *non*-coinjoin large-fan-in transaction instead.
        big_tx = Tx(
            "big_tx",
            inputs=[TxIO(f"W{i}", 10_000) for i in range(10)],
            outputs=[TxIO("Recipient", 90_000), TxIO("ChangeAddr", 9_500)],
        )
        txs_by_address = {f"W{i}": [big_tx] for i in range(10)}
        txs_by_address["Recipient"] = [big_tx]
        txs_by_address["ChangeAddr"] = [big_tx]
        source = FakeSource(txs_by_address, {"Recipient": 5, "ChangeAddr": 1})

        config = CrawlConfig(max_addresses=100, max_inputs_for_h1=5)  # cap below 10 inputs
        uf, stats = crawl_and_cluster(["W0"], source, config)
        self.assertFalse(uf.connected("W0", "W1"))
        self.assertEqual(stats.h1_merges, 0)

        config2 = CrawlConfig(max_addresses=100, max_inputs_for_h1=20)  # cap above 10 inputs
        uf2, stats2 = crawl_and_cluster(["W0"], source, config2)
        self.assertTrue(uf2.connected("W0", "W1"))
        self.assertGreater(stats2.h1_merges, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
