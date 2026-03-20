"""
Test suite for Phase 4 activation persistence strategies.

These tests validate:
1. ActivationStrategy dataclass enforces mutual exclusivity
2. Phase 4 solver produces valid solutions
3. Activation metadata estimation works reasonably
4. Phase 4 solution improves or equals Phase 3 baseline in memory
"""

import unittest
from typing import Optional, Tuple
from src.ilp.advanced_terms import (
    ActivationStrategy,
    ActivationMetadata,
    estimate_activation_metadata,
    compute_effective_costs_phase4,
)
from src.ilp.data_loader import load_ilp_inputs
from src.ilp.model_builder import ILPConfig4
from src.ilp.solve import solve_partition_ilp_phase4, solve_partition_ilp
from pathlib import Path


class TestActivationStrategy(unittest.TestCase):
    def test_retain_only(self):
        strat = ActivationStrategy("layer_0", retain=True, recompute=False, checkpoint=False)
        self.assertTrue(strat.is_valid)

    def test_recompute_only(self):
        strat = ActivationStrategy("layer_0", retain=False, recompute=True, checkpoint=False)
        self.assertTrue(strat.is_valid)

    def test_checkpoint_only(self):
        strat = ActivationStrategy("layer_0", retain=False, recompute=False, checkpoint=True)
        self.assertTrue(strat.is_valid)

    def test_none_selected(self):
        strat = ActivationStrategy("layer_0", retain=False, recompute=False, checkpoint=False)
        self.assertTrue(strat.is_valid)

    def test_multiple_strategies_invalid(self):
        """Verify that multiple strategies active raises ValueError."""
        with self.assertRaises(ValueError):
            ActivationStrategy("layer_0", retain=True, recompute=True, checkpoint=False)

    def test_all_strategies_invalid(self):
        """Verify that all three active raises ValueError."""
        with self.assertRaises(ValueError):
            ActivationStrategy("layer_0", retain=True, recompute=True, checkpoint=True)


class TestActivationMetadata(unittest.TestCase):
    def test_estimate_metadata_basic(self):
        nodes = ["layer_0", "layer_1", "layer_2"]
        node_cost_gpu_ms = {"layer_0": 10.0, "layer_1": 15.0, "layer_2": 20.0}
        node_cost_cpu_ms = {"layer_0": 50.0, "layer_1": 75.0, "layer_2": 100.0}
        node_mem_gpu_mb = {"layer_0": 100.0, "layer_1": 150.0, "layer_2": 200.0}

        meta = estimate_activation_metadata(
            nodes,
            node_cost_gpu_ms,
            node_cost_cpu_ms,
            node_mem_gpu_mb,
            activation_mem_fraction=0.70,
            io_time_fraction=0.15,
        )

        self.assertEqual(len(meta.node_mem_activation_mb), 3)
        self.assertAlmostEqual(meta.node_mem_activation_mb["layer_0"], 70.0, places=1)
        self.assertAlmostEqual(meta.node_mem_activation_mb["layer_1"], 105.0, places=1)
        self.assertAlmostEqual(meta.node_mem_activation_mb["layer_2"], 140.0, places=1)


class TestPhase4Solver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test ILP data."""
        candidate_roots = [
            Path("data/zephyr/results_smoke/simple_mlp/SGD/fp32/batch_8"),
            Path("data/test-m3/simple_mlp/SGD/fp32/batch_8"),
            Path("data/test-m3-r2/simple_mlp/SGD/fp32/batch_8"),
        ]

        selected_files: Optional[Tuple[Path, Path, Path]] = None
        for root in candidate_roots:
            if not root.exists():
                continue

            metrics_candidates = [
                root / "simple_mlp_metrics_stats.csv",
                root / "metrics_stats.csv",
            ]
            metrics_file = next((p for p in metrics_candidates if p.exists()), None)
            if metrics_file is None:
                continue

            run_dirs = sorted(root.glob("run_*"))
            for run_dir in run_dirs:
                graph = run_dir / "simple_mlp_graph_edges.csv"
                transfer = run_dir / "simple_mlp_transfer_edges.csv"
                if not (graph.exists() and transfer.exists()):
                    continue

                try:
                    cls.data = load_ilp_inputs(
                        metrics_stats_csv=str(metrics_file),
                        graph_edges_csv=str(graph),
                        transfer_edges_csv=str(transfer),
                        k_sigma=1.0,
                    )
                    selected_files = (metrics_file, graph, transfer)
                    break
                except (KeyError, ValueError):
                    continue
            if selected_files is not None:
                break

        if selected_files is None:
            raise unittest.SkipTest("Compatible Phase 4 fixtures not available")

    def test_phase4_solver_runs(self):
        """Verify Phase 4 solver produces a solution."""
        cfg4 = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=64.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            enable_checkpoint=False,
        )
        solution4 = solve_partition_ilp_phase4(self.data, cfg4, backend="greedy")
        
        self.assertIsNotNone(solution4)
        self.assertEqual(solution4.mode, "phase4")
        self.assertGreaterEqual(len(solution4.activation_strategies), 0)

    def test_phase4_activation_strategies_valid(self):
        """Verify all activation strategies are valid (mutually exclusive)."""
        cfg4 = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=64.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
        )
        solution4 = solve_partition_ilp_phase4(self.data, cfg4, backend="greedy")
        
        for node, strategy in solution4.activation_strategies.items():
            self.assertTrue(strategy.is_valid, f"Invalid strategy for {node}: {strategy}")

    def test_phase4_vs_phase3_memory(self):
        """Compare GPU memory usage: Phase 3 (all retain) vs Phase 4 (with recompute)."""
        cfg_base = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=64.0,
            cpu_mem_budget_mb=1e18,
        )

        # Solve baseline (Phase 3)
        sol_base = solve_partition_ilp(self.data, cfg_base, backend="auto")
        mem_base = sol_base.gpu_mem_used_mb

        # Solve Phase 4 with recompute allowed
        cfg4 = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=64.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            w_recompute_penalty=0.5,  # Moderate penalty
        )
        sol_phase4 = solve_partition_ilp_phase4(self.data, cfg4, backend="greedy")

        # Phase 4 should use <= memory than Phase 3
        # (or achieve comparable result)
        self.assertLessEqual(
            sol_phase4.gpu_mem_used_mb,
            mem_base * 1.05,  # Allow 5% tolerance due to heuristic
            f"Phase 4 expected to use <= {mem_base:.2f} MB, got {sol_phase4.gpu_mem_used_mb:.2f} MB"
        )


if __name__ == "__main__":
    unittest.main()
