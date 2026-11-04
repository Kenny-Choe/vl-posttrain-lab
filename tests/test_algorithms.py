import math
import unittest

from vl_posttrain.algorithms.objectives import advantages, surrogate, k3, dpo, gradient_probe
from vl_posttrain.algorithms.replay import FrozenRolloutBatch, verify_checkpoint_sidecar


class ObjectivesTests(unittest.TestCase):
    def test_diagnostics_keep_entropy_distinct_and_count_sign_clipping(self):
        from vl_posttrain.algorithms.diagnostics import token_diagnostics
        result=token_diagnostics([[-2+math.log(1.4),-2+math.log(.6)]],[[-2.,-2.]],[[-2.,-2.]],[1.],[[1,1]])
        self.assertEqual(result["ordinary_outside_clip_band_fraction"],1.)
        self.assertEqual(result["sign_effective_clip_fraction"],.5)
        self.assertIsNone(result["mean_full_distribution_entropy"])
        self.assertGreater(result["mean_sapo_gate_derivative_on_grpo_clipped_positions"],.9)

    def test_sapo_matches_grpo_first_on_policy_gradient(self):
        for a in (-1., 0., 1.):
            self.assertAlmostEqual(surrogate(-2., -2., a, "sapo")["d_loss_d_logp"],
                                   surrogate(-2., -2., a, "grpo")["d_loss_d_logp"])

    def test_sign_dependent_clipping(self):
        for a, ratio, clipped in [(1., 1.4, True), (-1., .6, True), (1., .6, False), (-1., 1.4, False)]:
            self.assertEqual(surrogate(math.log(ratio), 0., a, "grpo")["effective_clip"], clipped)

    def test_sapo_soft_gradient_survives_moderate_off_policy(self):
        self.assertLess(surrogate(math.log(1.35), 0., .5)["d_loss_d_logp"], -.6)
        self.assertGreater(surrogate(math.log(.65), 0., -.5)["d_loss_d_logp"], .3)

    def test_gradient_central_difference(self):
        self.assertEqual(gradient_probe()["status"], "passed")

    def test_advantage_population_std(self):
        values = advantages([0, 0, 1, 1])
        self.assertAlmostEqual(values[0], -1, places=7)
        self.assertAlmostEqual(sum(values), 0)

    def test_zero_reward_does_not_create_task_gradient(self):
        self.assertEqual(advantages([0] * 8), [0] * 8)
        self.assertEqual(surrogate(0, 0, 0)["d_loss_d_logp"], 0)

    def test_small_distance_coefficient_does_not_bound_normalized_advantage(self):
        high = advantages([.01] * 4 + [.04] * 4)
        low = advantages([.001] * 4 + [.004] * 4)
        self.assertAlmostEqual(high[0], -1, places=5)
        self.assertLess(abs(high[0] - low[0]), 1e-5)
        self.assertAlmostEqual(low[0], -.0015 / (.0015 + 1e-8), places=12)

    def test_kl_derivative_and_nonnegative_value(self):
        for current in (-5., -2., -.1):
            v = k3(current, -2.)
            step = 1e-6
            numerical = (k3(current + step, -2.)["value"] - k3(current - step, -2.)["value"]) / (2 * step)
            self.assertGreaterEqual(v["value"], 0)
            self.assertAlmostEqual(v["d_loss_d_logp"], numerical, places=6)

    def test_dpo_identical_initial_model_still_has_gradient(self):
        value = dpo(-2., -4., -2., -4.)
        self.assertAlmostEqual(value["loss"], math.log(2))
        self.assertEqual(value["d_loss_d_policy_chosen"], -.05)

    def test_replay_locks_old_ref_advantage_and_reuse_limit(self):
        rows = [{"sample_id":"q", "token_ids":[1], "old_logps":[-.5], "ref_logps":[-.5],
                 "advantage":0., "action_mask":[1], "prefix_sha256":"prefix"}]
        batch = FrozenRolloutBatch("old", "cold", rows)
        for reuse in range(1, 5):
            self.assertEqual(batch.next_reuse()["reuse"], reuse)
        with self.assertRaises(ValueError):
            batch.next_reuse()
        rows[0]["old_logps"][0] = -.6
        with self.assertRaises(ValueError):
            batch.next_reuse()

    def test_resume_cost_cannot_roll_back_or_lose_midbatch_rollout(self):
        sidecar = {"policy_sha256":"a", "ref_sha256":"b", "optimizer_sha256":"c", "scheduler_sha256":"d",
                   "rng_sha256":"e", "logical_batch_index":2, "reuse_index":1, "schedule_sha256":"f", "cost":{"generations":10}}
        with self.assertRaises(ValueError):
            verify_checkpoint_sidecar(sidecar)
        sidecar["frozen_rollout_sha256"] = "g"
        with self.assertRaises(ValueError):
            verify_checkpoint_sidecar(sidecar, {"generations":11})
        self.assertFalse(verify_checkpoint_sidecar(sidecar)["native_8b_resume_tested"])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "可选 Torch 未安装；不能将数学检查冒称原生可微测试")
class TorchTests(unittest.TestCase):
    def test_differentiable_loss_matches_scalar_and_mask(self):
        from vl_posttrain.algorithms.torch_objectives import policy_loss
        cur = torch.tensor([[math.log(1.35), float("nan")]], requires_grad=True)
        old = torch.zeros_like(cur)
        ref = torch.zeros_like(cur)
        loss, _ = policy_loss(cur, old, ref, torch.tensor([.5]), torch.tensor([[1, 0]]), beta=0.)
        loss.backward()
        self.assertAlmostEqual(cur.grad[0, 0].item(), surrogate(math.log(1.35), 0., .5)["d_loss_d_logp"], places=5)
        self.assertEqual(cur.grad[0, 1].item(), 0.)

    def test_chunked_causal_targets_match_cross_entropy_and_gradients(self):
        from vl_posttrain.algorithms.torch_objectives import chunked_target_logps
        logits = torch.randn(2, 5, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, 1, 5, -100, 8]])
        values, _ = chunked_target_logps(logits, labels, 2)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(-values.mean(), loss))
