#!/usr/bin/env python3
"""Offline compatibility-shell tests; fake CLI is confined to a temporary PATH."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'web/install-agent-skills.sh'


class InstallerCompatibility(unittest.TestCase):
    def test_supported_routes_delegate_to_one_cli(self):
        for option, expected in [([], ['universal', 'claude-code']), (['--cursor-only'], ['cursor']),
                                 (['--codex-only'], ['codex']), (['--claude-only'], ['claude-code'])]:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                executable = root / 'basilica'
                executable.write_text('#!/bin/bash\nif [[ "$2" == list ]]; then echo "Bundle version: pinned-test-version"; else printf "%s\\n" "$@" > "$BASILICA_TEST_ARGUMENTS"; fi\n')
                executable.chmod(0o755)
                env = {**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}',
                       'BASILICA_TEST_ARGUMENTS': str(root / 'args')}
                env.pop('BASILICA_AGENT_BASE_URL', None)
                subprocess.run(['bash', str(SCRIPT), *option], env=env, check=True)
                args = (root / 'args').read_text().splitlines()
                self.assertEqual(args[:3], ['skills', 'install', '-y'])
                self.assertEqual(args[3:], [word for agent in expected for word in ['--agent', agent]])

    def test_old_cli_and_retired_source_options_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'basilica'
            executable.write_text('#!/bin/bash\necho "Available skills: use-basilica"\n')
            executable.chmod(0o755)
            env = {**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}'}
            env.pop('BASILICA_AGENT_BASE_URL', None)
            result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('predates versioned skill bundles', result.stderr)
            result = subprocess.run(['bash', str(SCRIPT), '--base-url', 'https://example.invalid'], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no longer supported', result.stderr)


if __name__ == '__main__':
    unittest.main()
