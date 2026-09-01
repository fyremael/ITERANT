import unittest

from phase_rlvr import (
    HystereticEstimator,
    Phase,
    Telemetry,
    classify,
    length_normalized_policy_loss,
    pass_at_k_unbiased,
    recipe_for,
)


def telem(**overrides):
    base = dict(
        pass1=0.30,
        passk=0.55,
        pass1_slope=0.0,
        passk_slope=0.0,
        all_fail_rate=0.40,
        entropy=1.5,
        entropy_flow=0.0,
        correct_diversity=0.8,
        correct_diversity_slope=0.0,
        token_efficiency=0.02,
        token_efficiency_slope=0.0,
        kl=0.03,
        train_infer_gap=0.01,
        verifier_invariance=0.99,
    )
    base.update(overrides)
    return Telemetry(**base)


class PhaseContractTests(unittest.TestCase):
    def test_discovery(self):
        x = telem(pass1_slope=0.003, passk_slope=0.006)
        self.assertEqual(classify(x), Phase.DISCOVERY)
        self.assertEqual(recipe_for(Phase.DISCOVERY, x).name, "discovery")

    def test_sharpening(self):
        x = telem(pass1_slope=0.006, passk_slope=0.0002)
        self.assertEqual(classify(x), Phase.SHARPENING)
        self.assertEqual(recipe_for(Phase.SHARPENING, x).name, "sharpening")

    def test_stall(self):
        x = telem(pass1_slope=0.0001, passk_slope=-0.0002, token_efficiency_slope=0.0001)
        self.assertEqual(classify(x), Phase.STALL)
        self.assertEqual(recipe_for(Phase.STALL, x).name, "rebalance")

    def test_collapse_is_fail_closed(self):
        x = telem(passk_slope=0.02, entropy_flow=-0.05)
        self.assertEqual(classify(x), Phase.COLLAPSE)
        self.assertEqual(recipe_for(Phase.COLLAPSE, x).name, "recovery")

    def test_verifier_failure_overrides_apparent_progress(self):
        x = telem(pass1_slope=0.01, passk_slope=0.01, verifier_invariance=0.80)
        self.assertEqual(classify(x), Phase.COLLAPSE)

    def test_hysteresis_requires_confirmation(self):
        est = HystereticEstimator(confirmations=3, min_dwell=2)
        x = telem(pass1_slope=0.004, passk_slope=0.006)
        self.assertEqual(est.update(x), Phase.UNCERTAIN)
        self.assertEqual(est.update(x), Phase.UNCERTAIN)
        self.assertEqual(est.update(x), Phase.DISCOVERY)

    def test_collapse_bypasses_hysteresis(self):
        est = HystereticEstimator(confirmations=5, min_dwell=5)
        x = telem(kl=0.5)
        self.assertEqual(est.update(x), Phase.COLLAPSE)

    def test_pass_at_k(self):
        self.assertAlmostEqual(pass_at_k_unbiased(10, 1, 1), 0.1)
        self.assertGreater(pass_at_k_unbiased(10, 1, 5), 0.1)
        self.assertEqual(pass_at_k_unbiased(10, 6, 5), 1.0)

    def test_alpha_loss_contract(self):
        terms = [1.0, 1.0, 1.0, 1.0]
        self.assertEqual(length_normalized_policy_loss(terms, 2.0, 0.0), -8.0)
        self.assertEqual(length_normalized_policy_loss(terms, 2.0, 1.0), -2.0)
        self.assertAlmostEqual(length_normalized_policy_loss(terms, 2.0, 0.5), -4.0)

    def test_invalid_pass_relation_rejected(self):
        x = telem(pass1=0.8, passk=0.7)
        with self.assertRaises(ValueError):
            classify(x)


if __name__ == "__main__":
    unittest.main()
