import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def cli_defaults(relative_path: str) -> dict[str, object]:
    """Extract literal argparse defaults without importing GPU dependencies."""
    tree = ast.parse((ROOT / relative_path).read_text())
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[flag] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    pass
                break
    return defaults


def paper_config() -> dict[tuple[str, ...], object]:
    """Parse the scalar-only paper config with the standard library."""
    parsed: dict[tuple[str, ...], object] = {}
    parents: list[tuple[int, str]] = []
    for raw_line in (ROOT / "configs" / "paper.yaml").read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        key, separator, raw_value = raw_line.strip().partition(":")
        if not separator:
            continue
        while parents and parents[-1][0] >= indent:
            parents.pop()
        if not raw_value.strip():
            parents.append((indent, key))
            continue
        value_text = raw_value.strip()
        try:
            value = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            value = value_text
        parsed[tuple(name for _, name in parents) + (key,)] = value
    return parsed


class PaperDefaultsTest(unittest.TestCase):
    def assert_defaults(self, path: str, expected: dict[str, object]) -> None:
        actual = cli_defaults(path)
        for flag, value in expected.items():
            self.assertIn(flag, actual, f"{path}: {flag}")
            self.assertEqual(actual[flag], value, f"{path}: {flag}")

    def test_text_and_cross_modal_entry_points(self):
        self.assert_defaults("scripts/bench_lb.py", {
            "--K": 2, "--asym_method": "jsd", "--beta": 1.0,
            "--gamma": 0.5, "--slm": "4B", "--main_context": "summary",
        })
        self.assert_defaults("scripts/bench_multichallenge.py", {
            "--K": 2, "--asym_method": "jsd", "--beta": 1.0,
            "--gamma": 0.5, "--slm": "4B", "--n": 271,
            "--main_context": "summary_last_k", "--last_k": 1,
        })
        self.assert_defaults("scripts/bench_apibank.py", {
            "--K": 2, "--asym_method": "jsd", "--beta": 1.0,
            "--gamma": 0.5, "--slm": "1.7B", "--max_new": 256,
            "--main_compression": "name_sig",
        })
        self.assert_defaults("scripts/bench_mathvista.py", {
            "--K": 4, "--asym_method": "jsd", "--beta": 1.0,
            "--gamma": 0.5, "--n": 0,
        })

    def test_agentic_entry_points(self):
        common = {
            "--K": 2, "--asym_method": "jsd", "--beta": 1.0,
            "--gamma": 0.5, "--main_compression": "llmlingua",
            "--llmlingua_rate": 0.3, "--keep_last_k": 2,
        }
        self.assert_defaults("scripts/asym_smolagents/run_gaia_web.py", common)
        self.assert_defaults("scripts/asym_smolagents/run_simpleqa.py", {
            **common, "--n": 500,
        })

        source = (ROOT / "scripts/asym_smolagents/run_simpleqa.py").read_text()
        self.assertNotIn('asym_method="cma_vnorm"', source)
        self.assertGreaterEqual(source.count("asym_method=args.asym_method"), 1)

        for path in ["scripts/bench_gaia_full.py", "scripts/bench_gaia_vl.py"]:
            self.assert_defaults(path, {
                "--K": 2, "--asym_method": "jsd", "--beta": 1.0,
                "--gamma": 0.5,
            })

    def test_cross_family_entry_point(self):
        self.assert_defaults("experiments/cross_family/bench_lb_crossfamily.py", {
            "--K": 2, "--asym_method": "jsd", "--beta": 1.0,
            "--gamma": 0.5, "--main_context": "summary",
        })

    def test_config_matches_paper_defaults(self):
        config = paper_config()
        expected = {
            ("decoding", "temperature"): 0.0,
            ("decoding", "beta"): 1.0,
            ("decoding", "gamma"): 0.5,
            ("decoding", "cda_method"): "jsd",
            ("benchmarks", "longbench", "K"): 2,
            ("benchmarks", "multichallenge", "K"): 2,
            ("benchmarks", "multichallenge", "n"): 271,
            ("benchmarks", "api_bank", "K"): 2,
            ("benchmarks", "mathvista", "K"): 4,
            ("benchmarks", "gaia", "K"): 2,
            ("benchmarks", "simpleqa", "K"): 2,
            ("benchmarks", "simpleqa", "n"): 500,
        }
        for key, value in expected.items():
            self.assertEqual(config.get(key), value, ".".join(key))

    def test_internal_fallbacks_match_paper_defaults(self):
        files = [
            "vllm_specsteer/vllm_0_28/specsteer_sampler.py",
            "vllm_specsteer/vllm_0_28/specsteer_model.py",
            "vllm_specsteer/vllm_0_28/speculative.py",
        ]
        for relative_path in files:
            source = (ROOT / relative_path).read_text()
            self.assertNotIn("specsteer_gamma: float = 0.6", source)
            self.assertNotIn('"specsteer_gamma", 0.6', source)
            self.assertNotIn("gamma: float = 0.6", source)

        for relative_path in [
            "scripts/bench_mathvista.py",
            "scripts/bench_gaia_vl.py",
            "scripts/asym_smolagents/run_simpleqa.py",
        ]:
            source = (ROOT / relative_path).read_text()
            self.assertNotIn('asym_method="gamma_rule"', source)
            self.assertNotIn('asym_method="cma_vnorm"', source)

    def test_readme_representative_commands_use_paper_method(self):
        readme = (ROOT / "README.md").read_text()
        self.assertGreaterEqual(readme.count("--asym_method jsd"), 4)
        self.assertNotIn("--asym_method cma_vnorm", readme)


if __name__ == "__main__":
    unittest.main()
