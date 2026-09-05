"""Hermetic checker regressions; Docker/HTTP boundaries are isolated test doubles."""
import contextlib
import importlib.util
import io
import json
import os
import shutil
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'healthcheck.py'
SPEC = importlib.util.spec_from_file_location('localnet_healthcheck', SCRIPT)
healthcheck = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(healthcheck)


class HealthCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.selected = ['subtensor', 'validator', 'miner']
        self.missing = set()
        self.bad_header = set()
        self.chain = {'result': {'isSyncing': False, 'peers': 0}}
        self.calls = []
        self.config = {}
        # Use a real initialized SQLite fixture to validate the header contract.
        db = self.directory / 'database.sqlite'
        with contextlib.closing(sqlite3.connect(db)) as connection:
            connection.execute('create table checks (id integer primary key)')
            connection.commit()
        self.sqlite_header = db.read_bytes()[:15].decode()
        for service in ['validator', 'miner']:
            path = self.directory / f'{service}.toml'
            path.write_text(f'[database]\nurl = "sqlite:///custom/{service}.db"\n')
            self.config[service] = {'volumes': [
                {'type': 'bind', 'source': str(path), 'target': f'/app/{service}.toml'}]}

    def docker(self, *args):
        self.calls.append(args)
        if args[0] == 'compose':
            operation = args[3]
            if operation == 'config':
                return json.dumps({'services': {name: self.config.get(name, {}) for name in self.selected}})
            if operation == 'ps':
                service = args[-1]
                return '' if service in self.missing else f'{service}-container'
            if operation == 'port':
                return '127.0.0.1:39944'
        if args[0] == 'inspect':
            return json.dumps({'Running': True})
        if args[0] == 'exec':
            service = args[1].removesuffix('-container')
            self.assertEqual(args[-1], f'/custom/{service}.db')
            return '' if service in self.bad_header else self.sqlite_header
        raise AssertionError(f'Unexpected Docker call {args}')

    def http(self, service, port, path, payload=None):
        if service == 'subtensor':
            return json.dumps(self.chain)
        if path == '/health':
            return '{"status":"healthy"}'
        return f'basilica_{service}_requests_total 0\n'

    def run_check(self, profile='all'):
        checker = healthcheck.LocalnetHealth(profile)
        with patch.object(checker, 'docker', side_effect=self.docker), \
             patch.object(checker, 'http', side_effect=self.http), \
             patch.object(healthcheck.socket, 'create_connection', return_value=contextlib.nullcontext()), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            result = checker.run()
        return result, output.getvalue()

    def test_selected_profiles_do_not_require_unselected_services(self):
        for profile, services in [('network', ['subtensor']),
                                  ('validator', ['subtensor', 'validator']),
                                  ('miner', ['subtensor', 'validator', 'miner']),
                                  ('all', ['subtensor', 'validator', 'miner'])]:
            with self.subTest(profile=profile):
                self.selected = services
                self.calls.clear()
                result, output = self.run_check(profile)
                self.assertEqual(result, 0, output)
                inspected = [call[-1] for call in self.calls if call[0] == 'inspect']
                self.assertEqual(inspected, [f'{name}-container' for name in services])
                self.assertNotIn('postgres', repr(self.calls))
                self.assertNotIn('8091', repr(self.calls))

    def test_missing_required_container_fails_without_hiding_other_checks(self):
        self.missing.add('validator')
        result, output = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn('FAIL validator: expected one running container', output)
        self.assertIn('OK SQLite initialized: /custom/miner.db', output)

    def test_syncing_or_failed_chain_rpc_fails(self):
        self.selected = ['subtensor']
        for response in [{'result': {'isSyncing': True}}, {'error': {'code': -32601}}]:
            with self.subTest(response=response):
                self.chain = response
                self.assertEqual(self.run_check('network')[0], 1)

    def test_missing_or_uninitialized_sqlite_file_fails(self):
        for service in ['validator', 'miner']:
            with self.subTest(service=service):
                self.bad_header = {service}
                result, output = self.run_check()
                self.assertEqual(result, 1)
                self.assertIn(f'{service}: configured SQLite file is missing or uninitialized', output)

    def test_wrong_database_type_is_not_silently_accepted(self):
        (self.directory / 'validator.toml').write_text('[database]\nurl = "postgres://localhost/db"\n')
        result, output = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn('configured database is not an absolute SQLite file', output)

    def test_unknown_compose_service_requires_new_health_contract(self):
        self.selected = ['unsupported']
        result, output = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn('No health contract', output)

    def test_invalid_profile_fails_before_docker(self):
        result = subprocess.run([sys.executable, str(SCRIPT), 'monitoring'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_start_accepts_build_before_or_after_profile(self):
        # Stop at the Docker boundary: no chain, wallets or services are created.
        start = self.directory / 'start.sh'
        shutil.copy2(SCRIPT.with_name('start.sh'), start)
        (self.directory / 'configs').mkdir()
        for service in ['validator', 'miner']:
            (self.directory / 'configs' / f'{service}.toml').write_text('[database]\n')
        (self.directory / 'ssh-keys').mkdir()
        (self.directory / 'ssh-keys/miner_node_key').write_text('unused test fixture')
        binary = self.directory / 'bin'
        binary.mkdir()
        for name in ['curl', 'nc']:
            tool = binary / name
            tool.write_text('#!/bin/sh\nexit 0\n')
            tool.chmod(0o755)
        docker = binary / 'docker'
        docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit 97\n')
        docker.chmod(0o755)
        log = self.directory / 'calls'
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], CALL_LOG=str(log))
        for args, selected in [(['--build', 'network'], 'network'),
                               (['network', '--build'], 'network'),
                               (['--build'], 'miner')]:
            with self.subTest(args=args):
                log.write_text('')
                result = subprocess.run(['bash', str(start), *args], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
                self.assertIn('Profile: ' + selected, result.stdout)
                self.assertIn('compose --profile network up -d --build', log.read_text())

    def test_published_port_is_discovered_and_wildcard_maps_to_loopback(self):
        checker = healthcheck.LocalnetHealth('network')
        with patch.object(checker, 'docker', return_value='0.0.0.0:39944\n[::]:39944'):
            self.assertEqual(checker.endpoint('subtensor', 9944), ('127.0.0.1', 39944))


if __name__ == '__main__':
    unittest.main()
