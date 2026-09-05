#!/usr/bin/env python3
"""Regression checks for instruction validation using isolated project fixtures."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('instruction_check', Path(__file__).with_name('check-agent-instructions.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InstructionValidation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'arbitrary-checkout-name'
        (self.root / 'docs').mkdir(parents=True)
        (self.root / 'crates/widget').mkdir(parents=True)
        (self.root / 'AGENTS.md').write_text('[guide](docs/DEVELOPMENT.md)\n')
        (self.root / 'CLAUDE.md').symlink_to('AGENTS.md')
        (self.root / 'Cargo.toml').write_text('[workspace]\nmembers = ["crates/widget"]\n')
        (self.root / 'crates/widget/Cargo.toml').write_text('[package]\nname = "widget"\n[features]\ndatabase-tests = []\n')
        (self.root / 'docs/DEVELOPMENT.md').write_text('`cargo test --locked -p widget --features database-tests`\n')

    def test_standalone_layout_accepts_valid_paths_and_features(self):
        self.assertEqual(module.check(self.root), [])

    def test_reports_missing_local_link(self):
        (self.root / 'AGENTS.md').write_text('[missing](wrong-parent/docs/DEVELOPMENT.md)')
        self.assertTrue(any('broken link' in error for error in module.check(self.root)))

    def test_reports_unknown_package_and_feature(self):
        guide = self.root / 'docs/DEVELOPMENT.md'
        guide.write_text('`cargo test -p widget --features integration`\n`cargo test -p absent`\n')
        errors = module.check(self.root)
        self.assertTrue(any('has no feature integration' in error for error in errors))
        self.assertTrue(any('unknown Cargo package absent' in error for error in errors))

    def test_reports_divergent_scoped_entry_point(self):
        (self.root / 'crates/widget/AGENTS.md').write_text('Widget rules')
        (self.root / 'crates/widget/CLAUDE.md').write_text('Separate widget rules')
        self.assertTrue(any('crates/widget/CLAUDE.md' in error for error in module.check(self.root)))

    def test_reports_divergent_entry_point(self):
        (self.root / 'CLAUDE.md').unlink()
        (self.root / 'CLAUDE.md').write_text('A separate policy')
        self.assertTrue(any('authoritative AGENTS.md' in error for error in module.check(self.root)))


if __name__ == '__main__':
    unittest.main()
