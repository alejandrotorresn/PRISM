"""
Integration test for Phase 4 end-to-end solver on synthetic ILP instance.
"""

import unittest
from src.ilp.data_loader import ILPInputData
from src.ilp.model_builder import ILPConfig, ILPConfig4
from src.ilp.solve import solve_partition_ilp, solve_partition_ilp_phase4


class TestPhase4Synthetic(unittest.TestCase):
    def setUp(self):
        """Create a synthetic small ILP instance for testing."""
        # Simple 3-layer network
        self.data = ILPInputData(
            nodes=["layer_0", "layer_1", "layer_2"],
            node_cost_gpu_ms={
                "layer_0": 10.0,
                "layer_1": 15.0,
                "layer_2": 20.0,
            },
            node_cost_cpu_ms={
                "layer_0": 50.0,
                "layer_1": 75.0,
                "layer_2": 100.0,
            },
            node_energy_gpu_j={
                "layer_0": 0.5,
                "layer_1": 0.75,
                "layer_2": 1.0,
            },
            node_energy_cpu_j={
                "layer_0": 2.0,
                "layer_1": 3.0,
                "layer_2": 4.0,
            },
            node_mem_gpu_mb={
                "layer_0": 100.0,
                "layer_1": 150.0,
                "layer_2": 200.0,
            },
            node_mem_cpu_mb={
                "layer_0": 200.0,
                "layer_1": 300.0,
                "layer_2": 400.0,
            },
            edges=[
                ("layer_0", "layer_1"),
                ("layer_1", "layer_2"),
            ],
            edge_transfer_ms={
                ("layer_0", "layer_1"): 5.0,
                ("layer_1", "layer_2"): 7.0,
            },
            node_mem_activation_mb={
                "layer_0": 70.0,
                "layer_1": 105.0,
                "layer_2": 140.0,
            },
            node_time_io_ms={
                "layer_0": 1.5,
                "layer_1": 2.25,
                "layer_2": 3.0,
            },
            node_energy_io_j={
                "layer_0": 0.05,
                "layer_1": 0.05,
                "layer_2": 0.05,
            },
        )

    def test_phase4_synthetic_solver(self):
        """Test Phase 4 solver on synthetic data."""
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=350.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            w_recompute_penalty=0.5,
        )

        solution = solve_partition_ilp_phase4(self.data, cfg, backend="greedy")

        # Verify solution structure
        self.assertIsNotNone(solution)
        self.assertEqual(solution.mode, "phase4")
        self.assertEqual(solution.backend, "greedy_phase4")
        self.assertIn(solution.status, ["optimal", "feasible", "infeasible"])

        # Verify activation strategies
        self.assertEqual(len(solution.activation_strategies), 3)
        for node in self.data.nodes:
            self.assertIn(node, solution.activation_strategies)
            strat = solution.activation_strategies[node]
            self.assertTrue(strat.is_valid)

    def test_phase4_strategies_respect_budget(self):
        """Test that strategies respect GPU memory budget."""
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=350.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
        )

        solution = solve_partition_ilp_phase4(self.data, cfg, backend="greedy")

        # GPU memory should not exceed budget
        self.assertLessEqual(
            solution.gpu_mem_used_mb,
            cfg.gpu_mem_budget_mb * 1.01,  # Allow 1% tolerance for rounding
        )

    def test_phase4_all_nodes_have_strategies(self):
        """Test that all nodes have assigned strategies."""
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=350.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
        )

        solution = solve_partition_ilp_phase4(self.data, cfg, backend="greedy")

        # All nodes should have activation strategies
        for node in self.data.nodes:
            self.assertIn(node, solution.activation_strategies)

    def test_phase4_auto_selects_exhaustive_for_small_instances(self):
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=350.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
        )

        solution = solve_partition_ilp_phase4(self.data, cfg, backend="auto")
        self.assertEqual(solution.mode, "phase4")
        self.assertEqual(solution.backend, "exhaustive_phase4")
        self.assertEqual(len(solution.activation_strategies), len(self.data.nodes))

    def test_phase4_exhaustive_backend_runs(self):
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=350.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            enable_checkpoint=True,
            w_io=0.1,
        )

        solution = solve_partition_ilp_phase4(self.data, cfg, backend="exhaustive")
        self.assertEqual(solution.mode, "phase4")
        self.assertEqual(solution.backend, "exhaustive_phase4")
        self.assertIn(solution.status, ["optimal", "infeasible"])

    def test_base_solver_emits_dual_assignments(self):
        """The base ILP solver should now expose forward and backward assignments separately."""
        cfg = ILPConfig(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=350.0,
            cpu_mem_budget_mb=1e18,
        )

        solution = solve_partition_ilp(self.data, cfg, backend="exhaustive")

        self.assertIsNotNone(solution.forward_assignment)
        self.assertIsNotNone(solution.backward_assignment)
        forward_assignment = solution.forward_assignment or {}
        backward_assignment = solution.backward_assignment or {}
        self.assertEqual(set(forward_assignment.keys()), set(self.data.nodes))
        self.assertEqual(set(backward_assignment.keys()), set(self.data.nodes))
        self.assertEqual(solution.assignment, solution.forward_assignment)


if __name__ == "__main__":
    unittest.main()
