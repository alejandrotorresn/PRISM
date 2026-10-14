"""Regression test: _solve_phase4_greedy (src/ilp/solve.py) must offer
recompute/checkpoint candidates at WHOLE-BLOCK granularity for decoder-only
(torch.export) graphs, never a partial subset of a repeated block's leaves.

Why this exists
----------------
A decoder-only graph's "node" in the ILP is a single leaf submodule (e.g.
"transformer.h.0.ln_1"), decomposed from the real model at operator
granularity -- but the runtime (src/runtime/hybrid_executor.py) only ever
realizes real memory savings from recompute/checkpoint when ALL of a
repeated block's leaves share the same strategy together (confirmed
empirically, see fix/06 README): a single leaf alone saves ~0%. Before this
fix, the greedy heuristic scored and picked leaves one at a time, by their
OWN activation-memory/cost ratio, with no notion of "this leaf's block" --
it could (and, as this test's numbers are chosen to provoke, would) select
only the single most attractive leaf of a block while leaving its
siblings on retain, predicting a memory saving the runtime would never
actually deliver.
"""

import unittest

from src.ilp.data_loader import ILPInputData
from src.ilp.model_builder import ILPConfig4
from src.ilp.solve import solve_partition_ilp_phase4


def _decoder_like_graph() -> ILPInputData:
    # Two blocks of 3 leaves each, plus a standalone final layer (lm_head).
    # "ln_1" in each block is deliberately the single cheapest-to-recompute,
    # highest-activation leaf (best score under the OLD per-node heuristic),
    # so a correct, block-aware heuristic must either take the WHOLE block
    # (all 3 leaves) or none of it -- never "ln_1" alone.
    nodes = [
        "transformer.h.0.ln_1", "transformer.h.0.attn", "transformer.h.0.mlp",
        "transformer.h.1.ln_1", "transformer.h.1.attn", "transformer.h.1.mlp",
        "lm_head",
    ]
    cheap_gpu_ms = {n: 1.0 for n in nodes}
    expensive_cpu_ms = {n: 50.0 for n in nodes}
    base_mem = {n: 150.0 for n in nodes}
    act_mem = {
        "transformer.h.0.ln_1": 100.0, "transformer.h.0.attn": 40.0, "transformer.h.0.mlp": 40.0,
        "transformer.h.1.ln_1": 100.0, "transformer.h.1.attn": 40.0, "transformer.h.1.mlp": 40.0,
        "lm_head": 60.0,
    }
    recompute_io_ms = {
        # ln_1 looks artificially CHEAP to recompute alone (high score ==
        # act_mem / cost); its siblings are deliberately pricier so a
        # per-node heuristic would stop after ln_1 in each block.
        "transformer.h.0.ln_1": 0.5, "transformer.h.0.attn": 8.0, "transformer.h.0.mlp": 8.0,
        "transformer.h.1.ln_1": 0.5, "transformer.h.1.attn": 8.0, "transformer.h.1.mlp": 8.0,
        "lm_head": 3.0,
    }
    edges = [
        ("transformer.h.0.ln_1", "transformer.h.0.attn"),
        ("transformer.h.0.attn", "transformer.h.0.mlp"),
        ("transformer.h.0.mlp", "transformer.h.1.ln_1"),
        ("transformer.h.1.ln_1", "transformer.h.1.attn"),
        ("transformer.h.1.attn", "transformer.h.1.mlp"),
        ("transformer.h.1.mlp", "lm_head"),
    ]
    return ILPInputData(
        nodes=nodes,
        node_cost_gpu_ms=dict(cheap_gpu_ms),
        node_cost_cpu_ms=dict(expensive_cpu_ms),
        node_energy_gpu_j={n: 0.1 for n in nodes},
        node_energy_cpu_j={n: 1.0 for n in nodes},
        node_mem_gpu_mb=dict(base_mem),
        node_mem_cpu_mb={n: v * 2 for n, v in base_mem.items()},
        edges=edges,
        edge_transfer_ms={e: 1.0 for e in edges},
        node_mem_activation_mb=dict(act_mem),
        node_time_io_ms=dict(recompute_io_ms),
        node_energy_io_j={n: 0.01 for n in nodes},
        graph_trace_source="torch_export_decoder_only",
    )


class TestPhase4DecoderBlockGrouping(unittest.TestCase):
    def _run(self, graph_trace_source: str):
        data = _decoder_like_graph()
        data.graph_trace_source = graph_trace_source
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=200.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            target_gpu_utilization=0.5,
        )
        return solve_partition_ilp_phase4(data, cfg, backend="greedy")

    def test_decoder_only_graph_selects_whole_blocks_not_partial_leaves(self):
        solution = self._run("torch_export_decoder_only")
        self.assertEqual(solution.mode, "phase4")

        for prefix in ("transformer.h.0.", "transformer.h.1."):
            block_leaves = [n for n in solution.activation_strategies if n.startswith(prefix)]
            marked = [n for n in block_leaves if not solution.activation_strategies[n].retain]
            # Never a strict, non-empty subset: either every leaf in the
            # block got a strategy, or none did.
            self.assertIn(
                len(marked),
                (0, len(block_leaves)),
                f"block {prefix} got a PARTIAL strategy selection: {marked} of {block_leaves} -- "
                "the runtime gets zero benefit from this (see fix/06 README).",
            )

    def test_non_decoder_graph_also_requires_whole_block_grouping(self):
        # graph_trace_source="torch_fx" (module-granularity, vision-style):
        # fix/08 README found that a single isolated nn.Module gets zero
        # real benefit from checkpoint/recompute either (confirmed in
        # hybrid_executor.py's _detect_checkpoint_groups_call_module: +36%
        # WORSE in a controlled test) -- so this path must apply the SAME
        # whole-block-or-nothing rule as the decoder-only path, never a
        # per-node selection. With this graph's baseline (only
        # "transformer.h.0.ln_1" lands on GPU in Phase 3, never a whole
        # block together), no candidate qualifies, so nothing is marked.
        solution = self._run("torch_fx")
        self.assertEqual(solution.mode, "phase4")
        marked = [n for n, s in solution.activation_strategies.items() if not s.retain]
        self.assertEqual(marked, [])

        for prefix in ("transformer.h.0.", "transformer.h.1."):
            block_leaves = [n for n in solution.activation_strategies if n.startswith(prefix)]
            block_marked = [n for n in block_leaves if not solution.activation_strategies[n].retain]
            self.assertIn(
                len(block_marked),
                (0, len(block_leaves)),
                f"block {prefix} got a PARTIAL strategy selection: {block_marked} of {block_leaves} -- "
                "the runtime gets zero benefit from this for vision graphs either (see fix/08 README).",
            )
        # lm_head is its own block of size 1 in this synthetic graph. Unlike
        # the decoder-only case, a torch_fx graph has no final-node/
        # loss-fusion mechanism to save a standalone leaf alone (see
        # fix/08 README), so it must never be marked here.
        self.assertTrue(solution.activation_strategies["lm_head"].retain)


if __name__ == "__main__":
    unittest.main()
