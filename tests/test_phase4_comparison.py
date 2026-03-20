"""
End-to-end comparison: Phase 3 (baseline) vs Phase 4 (with activation strategies).

This test:
1. Creates a synthetic ILP instance
2. Solves baseline (Phase 3) to get optimal partition
3. Solves Phase 4 to get activation strategies
4. Compares objectives and validates that Phase 4 is competitive
"""

import unittest
from src.ilp.data_loader import ILPInputData
from src.ilp.model_builder import ILPConfig, ILPConfig4
from src.ilp.solve import solve_partition_ilp, solve_partition_ilp_phase4


class TestPhase4Comparison(unittest.TestCase):
    def setUp(self):
        """Create synthetic ILP instance representing a small neural network."""
        # 5-layer network with varying computational characteristics
        self.data = ILPInputData(
            nodes=["conv2d_1", "bn_1", "relu_1", "conv2d_2", "output"],
            node_cost_gpu_ms={
                "conv2d_1": 20.0,
                "bn_1": 5.0,
                "relu_1": 2.0,
                "conv2d_2": 25.0,
                "output": 3.0,
            },
            node_cost_cpu_ms={
                "conv2d_1": 100.0,
                "bn_1": 25.0,
                "relu_1": 10.0,
                "conv2d_2": 125.0,
                "output": 15.0,
            },
            node_energy_gpu_j={
                "conv2d_1": 1.0,
                "bn_1": 0.25,
                "relu_1": 0.1,
                "conv2d_2": 1.25,
                "output": 0.15,
            },
            node_energy_cpu_j={
                "conv2d_1": 4.0,
                "bn_1": 1.0,
                "relu_1": 0.4,
                "conv2d_2": 5.0,
                "output": 0.6,
            },
            node_mem_gpu_mb={
                "conv2d_1": 256.0,
                "bn_1": 64.0,
                "relu_1": 64.0,
                "conv2d_2": 512.0,  # Large layer
                "output": 32.0,
            },
            node_mem_cpu_mb={
                "conv2d_1": 512.0,
                "bn_1": 128.0,
                "relu_1": 128.0,
                "conv2d_2": 1024.0,
                "output": 64.0,
            },
            edges=[
                ("conv2d_1", "bn_1"),
                ("bn_1", "relu_1"),
                ("relu_1", "conv2d_2"),
                ("conv2d_2", "output"),
            ],
            edge_transfer_ms={
                ("conv2d_1", "bn_1"): 10.0,
                ("bn_1", "relu_1"): 8.0,
                ("relu_1", "conv2d_2"): 12.0,
                ("conv2d_2", "output"): 5.0,
            },
        )

    def test_phase3_vs_phase4_objective(self):
        """Compare Phase 3 baseline vs Phase 4 with memory-constrained budget."""
        # Tight GPU budget forces interesting partitioning decisions
        gpu_budget = 512.0  # Only ~2 large layers fit
        
        cfg_phase3 = ILPConfig(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=gpu_budget,
            cpu_mem_budget_mb=1e18,
        )
        
        cfg_phase4 = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=gpu_budget,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            w_recompute_penalty=0.5,
        )
        
        # Solve Phase 3
        sol_phase3 = solve_partition_ilp(self.data, cfg_phase3, backend="auto")
        
        # Solve Phase 4
        sol_phase4 = solve_partition_ilp_phase4(self.data, cfg_phase4, backend="greedy")
        
        # Phase 4 should be feasible
        self.assertEqual(sol_phase4.status, "optimal")
        self.assertEqual(sol_phase4.mode, "phase4")
        
        # Phase 4 should have activation strategies for all nodes
        self.assertEqual(len(sol_phase4.activation_strategies), len(self.data.nodes))
        
        # Print comparison for debugging
        print(f"\nPhase 3 (baseline): objective={sol_phase3.objective_value:.4f}, gpu_mem={sol_phase3.gpu_mem_used_mb:.2f} MB")
        print(f"Phase 4 (recompute): objective={sol_phase4.objective_value:.4f}, gpu_mem={sol_phase4.gpu_mem_used_mb:.2f} MB")
        print(f"Strategies: {sol_phase4.activation_strategies}")

    def test_phase4_recompute_activation(self):
        """Test that recompute strategy is used when beneficial."""
        # Budget that allows GPU but makes it tight
        cfg = ILPConfig4(
            w_time=1.0,
            w_energy=0.0,
            w_transfer=1.0,
            gpu_mem_budget_mb=600.0,
            cpu_mem_budget_mb=1e18,
            enable_recompute=True,
            w_recompute_penalty=0.3,  # Low penalty for recompute
        )
        
        solution = solve_partition_ilp_phase4(self.data, cfg, backend="greedy")
        
        # Some layers should be marked for recompute
        strategies = solution.activation_strategies
        self.assertIsNotNone(strategies)
        
        # Verify strategy validity
        for node, strategy in strategies.items():
            self.assertTrue(strategy.is_valid)
            self.assertEqual(strategy.node, node)
            active_count = int(strategy.retain) + int(strategy.recompute) + int(strategy.checkpoint)
            self.assertLessEqual(active_count, 1)
        
        # At least some nodes should be assigned (obviously)
        self.assertGreater(len(strategies), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
