import json
import unittest

from humansense.feedback import CalibrationArtifact, LabeledOutcome, fit_calibration
from humansense.schema import validate_output


def samples(prefix, score, count, positives):
    return [LabeledOutcome(f"{prefix}-{i}", score, i < positives) for i in range(count)]


def fit(training, validation, **options):
    return fit_calibration(target="user_requests_assistance", training=training, validation=validation,
        training_dataset="train-session-v1", validation_dataset="heldout-session-v1",
        scorer_model_id="help-scorer", scorer_model_version="1", **options)


class CalibrationTests(unittest.TestCase):
    def test_train_frequencies_and_heldout_metrics(self):
        model = fit(samples("train", .85, 20, 16), samples("test", .85, 20, 16))
        result = model.predict(.81)
        self.assertAlmostEqual(result["probability"], .8)
        self.assertTrue(result["calibrated"])
        self.assertAlmostEqual(model.evaluation["ece"], 0)
        self.assertAlmostEqual(model.evaluation["brier_score"], .16)
        self.assertEqual(result["calibration"]["dataset"], "heldout-session-v1")
        validate_output(result)

    def test_insufficient_training_and_validation_abstain(self):
        for train_count, heldout_count in ((19, 20), (20, 19), (0, 0)):
            model = fit(samples("train", .85, train_count, 0), samples("test", .85, heldout_count, 0))
            result = model.predict(.81)
            self.assertIsNone(result["probability"])
            self.assertFalse(result["calibrated"])
            self.assertEqual(result["reason"], "insufficient_bin_samples")
            validate_output(result)

    def test_overlapping_and_duplicate_ids_rejected(self):
        point = LabeledOutcome("same", .85, True)
        for training, validation in (([point], [point]), ([point, point], []), ([], [point, point])):
            with self.assertRaisesRegex(ValueError, "leakage"):
                fit(training, validation)

    def test_heldout_labels_never_change_train_prediction(self):
        model = fit(samples("train", .85, 20, 16), samples("test", .85, 20, 0))
        self.assertEqual(model.bins[4].probability, .8)
        self.assertFalse(model.validated)
        result = model.predict(.85)
        self.assertEqual(result["reason"], "held_out_validation_failed")
        self.assertIsNone(result["probability"])

    def test_coverage_counts_unseen_or_unvalidated_bins(self):
        model = fit(samples("train", .85, 20, 16),
                    samples("test", .85, 20, 16)+samples("other", .15, 60, 0))
        self.assertAlmostEqual(model.evaluation["coverage"], .25)
        self.assertFalse(model.validated)
        self.assertIsNone(model.predict(.85)["probability"])

    def test_per_bin_not_global_support_required(self):
        model = fit(samples("train", .85, 20, 16), samples("test", .85, 20, 16))
        self.assertTrue(model.validated)
        self.assertIsNone(model.predict(.15)["probability"])
        self.assertIsNone(model.predict(.85, model_version="different")["probability"])

    def test_small_bad_bin_cannot_hide_behind_global_average(self):
        model = fit(samples("train-hi", .85, 100, 100)+samples("train-low", .15, 20, 0),
                    samples("test-hi", .85, 100, 100)+samples("test-low", .15, 20, 10))
        self.assertTrue(model.validated)
        self.assertLess(model.evaluation["ece"], .1)
        self.assertEqual(model.predict(.85)["probability"], 1)
        self.assertEqual(model.predict(.15)["reason"], "held_out_bin_validation_failed")
        self.assertIsNone(model.predict(.15)["probability"])

    def test_round_trip_and_tampered_counts(self):
        model = fit(samples("train", 1.0, 20, 20), samples("test", 1.0, 20, 20))
        loaded = CalibrationArtifact.from_json(model.to_json())
        self.assertEqual(loaded, model)
        self.assertEqual(loaded.predict(1.0)["probability"], 1)
        self.assertIsNone(loaded.predict(0.0)["probability"])
        changed = model.to_dict()
        changed["bins"][4]["training_positives"] = 21
        with self.assertRaises(ValueError):
            CalibrationArtifact.from_json(json.dumps(changed))

    def test_sample_types_and_configuration(self):
        for score in (float("nan"), float("inf"), -1, 1.1, True):
            with self.assertRaises(ValueError):
                LabeledOutcome("a", score, True)
        with self.assertRaises(ValueError):
            LabeledOutcome("a", .5, 1)
        with self.assertRaises(ValueError):
            fit([], [], bins=0)
        with self.assertRaises(ValueError):
            fit([], [], min_coverage=0)
        with self.assertRaises(ValueError):
            fit_calibration(target="help", training=[], validation=[], training_dataset="same",
                validation_dataset="same", scorer_model_id="a", scorer_model_version="1")


if __name__ == "__main__":
    unittest.main()
