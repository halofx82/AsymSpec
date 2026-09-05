from pathlib import Path
import ast
import os
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".txt"}


def release_files():
    for directory, subdirs, filenames in os.walk(ROOT):
        subdirs[:] = [
            name for name in subdirs
            if name not in {".git", "__pycache__", ".venv", ".cache", ".backups",
                            "data", "outputs", "artifacts", "results", "runs"}
        ]
        base = Path(directory)
        for filename in filenames:
            yield base / filename


class ReleaseTreeTest(unittest.TestCase):
    def test_required_release_files_exist(self):
        for name in [
            "README.md", "LICENSE", "NOTICE", "AUTHORS.md",
            "CITATION.cff", "RELEASE_PROVENANCE.md", "requirements.txt",
            "VLLM_COMPATIBILITY.md",
        ]:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_no_machine_specific_paths(self):
        forbidden = ["/root/dataDisk", "/mnt/disk0", "/mnt/src/", "PPTAgent/"]
        failures = []
        for path in release_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            for token in forbidden:
                if token in text:
                    failures.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_no_common_secret_shapes(self):
        patterns = [
            re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
            re.compile(r"hf_[A-Za-z0-9]{20,}"),
            re.compile(r"AKIA[0-9A-Z]{16}"),
            re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
        ]
        failures = []
        for path in release_files():
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            if any(pattern.search(text) for pattern in patterns):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_no_large_release_artifacts(self):
        large = [
            str(path.relative_to(ROOT))
            for path in release_files()
            if path.is_file()
            and path.stat().st_size > 5 * 1024 * 1024
        ]
        self.assertEqual(large, [])

    def test_no_internal_versioned_entry_points(self):
        path_patterns = [
            re.compile(r"(?:^|/)v0\.\d+(?:\.mm)?(?:/|$)"),
            re.compile(r"_v\d+(?:\.py)?$"),
            re.compile(r"benchmark\d+\.py$"),
        ]
        content_tokens = [
            "bench_mc_v07", "bench_apibank_v2", "run_benchmark100",
            "v0.10.mm", "Phase 2 Option Z", "HETERO_DESIGN.md",
        ]
        failures = []
        for path in release_files():
            relative = str(path.relative_to(ROOT))
            if any(pattern.search(relative) for pattern in path_patterns):
                failures.append(relative)
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            for token in content_tokens:
                if token in text:
                    failures.append(f"{relative}: {token}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_cross_family_patch_anchors_match_release_sources(self):
        patch_path = ROOT / "experiments/cross_family/apply_hetero_patch.py"
        tree = ast.parse(patch_path.read_text())
        values = {}
        edit_lists = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id.endswith(("_OLD", "_NEW")):
                values[target.id] = ast.literal_eval(node.value)
            elif target.id in {"model_edits", "sampler_edits"}:
                edit_lists[target.id] = [
                    (item.elts[1].id, item.elts[2].id)
                    for item in node.value.elts
                ]

        model = (ROOT / "vllm_specsteer/vllm_0_28/specsteer_model.py").read_text()
        sampler = (ROOT / "vllm_specsteer/vllm_0_28/specsteer_sampler.py").read_text()
        self.assertTrue(values)
        for list_name, source in [
            ("model_edits", model), ("sampler_edits", sampler),
        ]:
            for old_name, new_name in edit_lists[list_name]:
                anchor = values[old_name]
                self.assertEqual(source.count(anchor), 1, old_name)
                source = source.replace(anchor, values[new_name], 1)
            ast.parse(source)

        patch_source = patch_path.read_text()
        for required in [
            "E5 Gate-A bonus map",
            "F1 fast-path bonus mask",
            "F2 fast-path draft mask",
            "F3 fast-path draft map",
            "F4 heterogeneous base path",
            "bonus_in_range",
        ]:
            self.assertIn(required, patch_source)


if __name__ == "__main__":
    unittest.main()
