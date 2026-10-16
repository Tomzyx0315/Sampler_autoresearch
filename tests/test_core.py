"""Dependency-free checks for numerical updates and the experiment entry point."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from baselines.heun import sample as heun
from baselines.dpmpp import sample as dpmpp
from harness.common import schedule, read_json
from harness import evaluate


class CoreTests(unittest.TestCase):
    def test_analytic_updates_and_nfe(self):
        sigmas = schedule(10)
        for algorithm, expected in ((dpmpp, 10), (heun, 19)):
            calls = []
            def denoise(x, sigma):
                self.assertGreater(sigma, 0)
                calls.append(sigma)
                return 0.75
            self.assertAlmostEqual(algorithm(denoise, 23.0, sigmas), 0.75, places=12)
            self.assertEqual(len(calls), expected)
            # Constant velocity: exact endpoint x(0)=x(s)-s*v.
            actual = algorithm(lambda x, s: x - 0.125*s, 12.0, sigmas)
            self.assertAlmostEqual(actual, 12.0 - 0.125*sigmas[0], places=10)

    def test_gaussian_flow_converges(self):
        variance, initial = 2.0, 80.0
        exact = initial * math.sqrt(variance / (variance + 80**2))
        for algorithm in (dpmpp, heun):
            errors = [abs(algorithm(lambda x, s: x*variance/(variance+s*s), initial, schedule(n)) - exact)
                      for n in (16, 128)]
            self.assertLess(errors[1], errors[0] / 10)
            self.assertLess(errors[1], 0.005)

    def test_flexible_cli(self):
        args = evaluate.parse_args(["--num", "37", "--fid", "--trace", "3", "--sampler", "my_method:sample"])
        self.assertEqual((args.num, args.trace, args.sampler), (37, 3, "my_method:sample"))
        with self.assertRaises(ValueError):
            schedule(1)

    def test_failure_record_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            command = ["evaluate", "--run", "failure-test"]
            with mock.patch.object(evaluate, "ROOT", Path(folder)), mock.patch("sys.argv", command), \
                 mock.patch.object(evaluate, "evaluate", side_effect=RuntimeError("test error")):
                with self.assertRaisesRegex(RuntimeError, "test error"):
                    evaluate.main()
                result = Path(folder) / "experiments/failure-test/result.json"
                self.assertEqual(read_json(result)["status"], "failed")
                before = result.read_bytes()
                with self.assertRaises(FileExistsError):
                    evaluate.main()
                self.assertEqual(result.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
