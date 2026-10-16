"""Run with Python's standard library; no model downloads or dependencies."""
import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
from baselines.heun import sample as heun
from baselines.dpmpp import sample as dpmpp
from harness.common import (ROOT, schedule, run_id, write_json, read_json, sha256,
                            asset_paths, source_hashes, snapshot)
from harness.evaluate import parser, validate


class NumericalTests(unittest.TestCase):
    def test_constant_denoiser_and_nfe(self):
        sigmas = schedule(10)
        for algorithm, expected in ((dpmpp, 10), (heun, 19)):
            calls = []
            def denoise(x, sigma):
                self.assertGreater(sigma, 0)
                calls.append(sigma)
                return 0.75
            self.assertAlmostEqual(algorithm(denoise, 23.0, sigmas), 0.75, places=12)
            self.assertEqual(len(calls), expected)

    def test_constant_velocity(self):
        sigmas = schedule(7)
        # D=x-sigma*v makes dx/dsigma=v, so the exact endpoint is known.
        for algorithm in (dpmpp, heun):
            actual = algorithm(lambda x, s: x - 0.125 * s, 12.0, sigmas)
            self.assertAlmostEqual(actual, 12.0 - 0.125 * sigmas[0], places=10)

    def test_gaussian_flow_converges(self):
        # Exact Gaussian denoiser and ODE solution provide an independent oracle.
        variance, initial = 2.0, 80.0
        exact = initial * math.sqrt(variance / (variance + 80**2))
        for algorithm in (dpmpp, heun):
            errors = []
            for steps in (16, 128):
                actual = algorithm(lambda x, s: x * variance / (variance + s*s), initial, schedule(steps))
                errors.append(abs(actual - exact))
            self.assertLess(errors[1], errors[0] / 10)
            self.assertLess(errors[1], 0.005)

    def test_schedule_and_rejections(self):
        levels = schedule(10)
        self.assertEqual(len(levels), 11)
        self.assertAlmostEqual(levels[0], 80)
        self.assertAlmostEqual(levels[-2], 0.002)
        self.assertEqual(levels[-1], 0)
        self.assertTrue(all(a > b for a, b in zip(levels, levels[1:])))
        for args in ((1,), (5, -1), (5, 2, 1), (5, 0.002, 80, 0)):
            with self.assertRaises(ValueError):
                schedule(*args)


class RecordTests(unittest.TestCase):
    def test_run_lifecycle_preserves_failed_runs(self):
        from harness import evaluate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = ["evaluate", "--run", "failed-trial", "--hypothesis", "Exercise failure recording"]
            with mock.patch.object(evaluate, "ROOT", root), mock.patch("sys.argv", command), \
                 mock.patch.object(evaluate, "execute", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    evaluate.main()
                record = root / "experiments/failed-trial"
                status = read_json(record / "status.json")
                self.assertEqual(status["state"], "failed")
                self.assertIn("test failure", (record / "error.txt").read_text())
                self.assertFalse((record / "metrics.json").exists())
                self.assertTrue((root / "artifacts/failed-trial/source.zip").exists())
                before = (record / "status.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    evaluate.main()
                self.assertEqual((record / "status.json").read_bytes(), before)

    def test_successful_run_records_completion(self):
        from harness import evaluate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = ["evaluate", "--run", "lifecycle-test", "--hypothesis", "Test record lifecycle only"]
            with mock.patch.object(evaluate, "ROOT", root), mock.patch("sys.argv", command), \
                 mock.patch.object(evaluate, "execute", return_value={}):
                evaluate.main()
            record = root / "experiments/lifecycle-test"
            self.assertEqual(read_json(record / "status.json")["state"], "completed")
            manifest = read_json(record / "manifest.json")
            self.assertEqual(manifest["hypothesis"], "Test record lifecycle only")
            self.assertIn("sampler.py", manifest["source_hashes"])
            self.assertEqual(manifest["source_archive_sha256"], sha256(root / "artifacts/lifecycle-test/source.zip"))

    def test_atomic_json_and_nan_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "metrics.json"
            write_json(p, {"fid": 3.0})
            with self.assertRaises(ValueError):
                write_json(p, {"fid": float("nan")})
            self.assertEqual(read_json(p), {"fid": 3.0})

    def test_asset_tampering_is_detected(self):
        config = read_json(ROOT / "configs/cifar10.json")
        spec = config["assets"]["network"]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / spec["filename"]
            p.write_bytes(b"test asset")
            write_json(Path(folder) / "manifest.json", {"network": {**spec, "sha256": sha256(p)}})
            asset_paths(folder, ["network"])
            p.write_bytes(b"modified asset")
            with self.assertRaises(ValueError):
                asset_paths(folder, ["network"])

    def test_snapshot_captures_executable_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "source.zip"
            hashes = snapshot(p)
            self.assertEqual(hashes, source_hashes())
            with zipfile.ZipFile(p) as archive:
                for relative in hashes:
                    self.assertEqual(archive.read(relative), (ROOT / relative).read_bytes())
                self.assertIn("sampler.py", archive.namelist())
                self.assertIn("harness/evaluate.py", archive.namelist())

    def test_run_paths_cannot_escape(self):
        for value in ("../escape", "/tmp/escape", ".", "a/b", "a b", ""):
            with self.assertRaises(ValueError):
                run_id(value)
        self.assertEqual(run_id("exp-001_baseline"), "exp-001_baseline")

    def test_evaluation_arguments(self):
        def args(*extra):
            return parser().parse_args(["--run", "test", "--hypothesis", "test", *extra])
        self.assertEqual(validate(args())[0], "sampler:sample")
        self.assertEqual(validate(args("--sampler", "playground.example:sample"))[0], "playground.example:sample")
        for extra in (("--trace", "4", "--preset", "full"), ("--steps", "1"),
                      ("--sampler", "os:system"), ("--params", "[]"),
                      ("--params", '{"x": NaN}'), ("--max-seconds", "nan")):
            with self.assertRaises(ValueError):
                validate(args(*extra))

    def test_vendored_code_matches_manifest(self):
        for relative, expected in read_json(ROOT / "third_party/manifest.json").items():
            self.assertEqual(sha256(ROOT / "third_party" / relative), expected, relative)

    def test_python_sources_parse(self):
        for relative in source_hashes():
            if relative.endswith(".py"):
                ast.parse((ROOT / relative).read_text(), filename=relative)


if __name__ == "__main__":
    unittest.main()
