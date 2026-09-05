#!/usr/bin/env python3
"""Read-only health checks for the services selected by a localnet Compose profile."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parent
PROFILES = {'network': 'network', 'subtensor': 'network', 'validator': 'validator',
            'val': 'validator', 'miner': 'miner', 'min': 'miner', 'all': 'all'}


class CheckFailure(Exception):
    pass


class LocalnetHealth:
    def __init__(self, profile):
        self.compose = ['compose', '--profile', profile]
        self.failures = []

    def docker(self, *args):
        try:
            result = subprocess.run(['docker', *args], cwd=ROOT, check=True,
                                    capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # Do not echo Docker config/environment: it can contain credentials.
            raise CheckFailure(f'Docker command failed: {type(error).__name__}') from error
        return result.stdout.strip()

    def endpoint(self, service, port):
        published = self.docker(*self.compose, 'port', service, str(port)).splitlines()
        if not published:
            raise CheckFailure(f'{service}:{port} has no published host port')
        host, actual_port = published[0].rsplit(':', 1)
        host = host.strip('[]')
        if host in ('0.0.0.0', '::'):
            host = '127.0.0.1'
        return host, int(actual_port)

    def http(self, service, port, path, payload=None):
        host, actual_port = self.endpoint(service, port)
        address = f'[{host}]' if ':' in host else host
        request = urllib.request.Request(f'http://{address}:{actual_port}{path}')
        if payload is not None:
            request.data = json.dumps(payload).encode()
            request.add_header('Content-Type', 'application/json')
        # Local health requests must not be routed through ambient HTTP proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=5) as response:
            return response.read(1024 * 1024).decode()

    def sqlite(self, service, definition, container):
        mounted = next((volume for volume in definition.get('volumes', [])
                        if volume.get('target') == f'/app/{service}.toml'), None)
        if not mounted or mounted.get('type') != 'bind':
            raise CheckFailure(f'{service}: expected bind-mounted configuration')
        config = tomllib.loads(Path(mounted['source']).read_text())
        database_url = config.get('database', {}).get('url', '')
        if not database_url.startswith('sqlite:///'):
            raise CheckFailure(f'{service}: configured database is not an absolute SQLite file')
        database_path = database_url.removeprefix('sqlite://').split('?', 1)[0]
        # Read only the initialized SQLite header. No database writes, copying
        # live WAL files, sqlite CLI dependency, or integrity claims.
        header = self.docker('exec', container, 'head', '-c', '15', database_path)
        if header != 'SQLite format 3':
            raise CheckFailure(f'{service}: configured SQLite file is missing or uninitialized')
        print(f'  OK SQLite initialized: {database_path} (header check only)')

    def check(self, service, definition):
        container = self.docker(*self.compose, 'ps', '-q', service)
        if not container or '\n' in container:
            raise CheckFailure(f'{service}: expected one running container in this Compose project')
        state = json.loads(self.docker('inspect', '--format', '{{json .State}}', container))
        if not state.get('Running') or state.get('Health', {}).get('Status') in ('starting', 'unhealthy'):
            raise CheckFailure(f'{service}: container is not ready')
        if service == 'subtensor':
            response = json.loads(self.http(service, 9944, '/', {
                'jsonrpc': '2.0', 'id': 1, 'method': 'system_health', 'params': []}))
            health = response.get('result')
            if not isinstance(health, dict) or health.get('isSyncing') is not False:
                raise CheckFailure('Subtensor RPC unavailable or still syncing')
            print('  OK chain RPC responding and not syncing')
        elif service in ('validator', 'miner'):
            if service == 'validator':
                response = json.loads(self.http(service, 8080, '/health'))
                if response.get('status') != 'healthy':
                    raise CheckFailure('Validator health response is not healthy')
                print('  OK validator HTTP health')
            else:
                host, port = self.endpoint(service, 50051)
                with socket.create_connection((host, port), timeout=5):
                    pass
                print('  OK miner gRPC port accepting TCP (not an authenticated RPC test)')
            metrics = self.http(service, 9090, '/metrics')
            if f'basilica_{service}_' not in metrics:
                raise CheckFailure(f'{service}: metrics endpoint returned no service metrics')
            print('  OK service metrics')
            self.sqlite(service, definition, container)
        else:
            raise CheckFailure(f'No health contract for Compose service {service!r}; update the checker')

    def run(self):
        config = json.loads(self.docker(*self.compose, 'config', '--format', 'json'))
        services = config.get('services', {})
        if not services:
            raise CheckFailure('Selected profile contains no services')
        print('Selected services: ' + ', '.join(services))
        for service, definition in services.items():
            print(service + ':')
            try:
                self.check(service, definition)
            except (CheckFailure, OSError, ValueError, KeyError) as error:
                self.failures.append(service)
                print(f'  FAIL {error}')
        if self.failures:
            print('UNHEALTHY: ' + ', '.join(self.failures))
            return 1
        print('HEALTHY: all selected service checks passed; no end-to-end rental/training claim')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Uses the current Compose project/COMPOSE_FILE and published ports. '
        'Exit 0: selected checks pass; 1: unhealthy/unverifiable; 2: invalid arguments. '
        'Requires Python 3.11+ and Docker Compose. Does not create resources.'))
    parser.add_argument('profile', nargs='?', default='all', choices=sorted(PROFILES))
    args = parser.parse_args()
    # An ambient profile must not silently add services beyond the selected one.
    os.environ.pop('COMPOSE_PROFILES', None)
    try:
        return LocalnetHealth(PROFILES[args.profile]).run()
    except (CheckFailure, OSError, ValueError, KeyError) as error:
        print(f'UNVERIFIABLE: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
