"""Exercise deployment against disposable package trees, without importing vLLM."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts/deploy_specsteer.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pkg = self.root / "vllm"
        self.pkg.mkdir()
        self.payload = self.root / "payload"
        self.payload.mkdir()
        self.backup = self.root / "backups"
        (self.pkg / "existing.py").write_text("original\n")
        (self.payload / "existing.py").write_text("patched\n")
        (self.payload / "new.py").write_text("added\n")
        entries = [{"source": n, "target": n,
                    "upstream_sha256": deploy.sha256(self.pkg / n)}
                   for n in ("existing.py", "new.py")]
        (self.payload / "manifest.json").write_text(json.dumps({"files": entries}))
        p = patch.object(deploy, "PATCH_DIR", self.payload)
        p.start()
        self.addCleanup(p.stop)

    def test_apply_repeat_update_revert(self):
        deploy.apply(self.pkg, self.backup)
        deploy.apply(self.pkg, self.backup)
        (self.payload / "existing.py").write_text("updated port\n")
        deploy.apply(self.pkg, self.backup)
        self.assertEqual((self.pkg / "existing.py").read_text(), "updated port\n")
        deploy.revert(self.pkg, self.backup)
        self.assertEqual((self.pkg / "existing.py").read_text(), "original\n")
        self.assertFalse((self.pkg / "new.py").exists())
        deploy.apply(self.pkg, self.backup)
        deploy.revert(self.pkg, self.backup)

    def test_preflight_does_not_partially_deploy(self):
        (self.payload / "new.py").unlink()
        with self.assertRaises(RuntimeError):
            deploy.apply(self.pkg, self.backup)
        self.assertEqual((self.pkg / "existing.py").read_text(), "original\n")
        self.assertFalse(self.backup.exists())

    def test_refuse_modified_installation(self):
        (self.pkg / "existing.py").write_text("user edit\n")
        with self.assertRaises(RuntimeError):
            deploy.apply(self.pkg, self.backup)

    def test_refuse_wrong_version(self):
        with patch.object(deploy.importlib.metadata, "distribution") as dist:
            dist.return_value.version = "0.19.0"
            with self.assertRaises(RuntimeError):
                deploy.package_root()

    def test_preserve_post_deployment_edits_on_revert(self):
        deploy.apply(self.pkg, self.backup)
        (self.pkg / "new.py").write_text("user edit\n")
        with self.assertRaises(RuntimeError):
            deploy.revert(self.pkg, self.backup)
        self.assertEqual((self.pkg / "existing.py").read_text(), "patched\n")


if __name__ == "__main__":
    unittest.main()
