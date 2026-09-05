#!/usr/bin/env python3
"""Validate maintained instruction links and declared contributor Cargo checks.

This deliberately does not execute operator examples, contact services, or claim
that arbitrary prose is semantically correct. CLI parser tests cover commands.
"""
from pathlib import Path
import re
import hashlib
import json
import tarfile
import sys
import tomllib
from urllib.parse import unquote, urlsplit


def check(root: Path) -> list[str]:
    errors = []
    documents = [root / 'AGENTS.md', root / 'docs/DEVELOPMENT.md']
    documents += sorted((root / 'crates').glob('*/AGENTS.md'))
    documents += sorted((root / '.claude/skills').glob('*/SKILL.md'))
    distribution = root / 'docs/AGENT-SKILLS.md'
    if distribution.exists():
        documents.append(distribution)
    for document in documents:
        if not document.is_file():
            errors.append(f'{document.relative_to(root)}: missing instruction entry point')
            continue
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            parsed = urlsplit(link)
            if parsed.scheme or not parsed.path:
                continue
            target = (document.parent / unquote(parsed.path)).resolve()
            if not target.exists():
                errors.append(f'{document.relative_to(root)}: broken link {link}')
    for agent_file in [root / 'AGENTS.md', *sorted((root / 'crates').glob('*/AGENTS.md'))]:
        claude = agent_file.with_name('CLAUDE.md')
        if agent_file.parent != root and not claude.exists() and not claude.is_symlink():
            continue
        if not claude.is_symlink() or claude.resolve() != agent_file.resolve():
            errors.append(f'{claude.relative_to(root)} must link to the authoritative AGENTS.md')
    manifest = tomllib.loads((root / 'Cargo.toml').read_text())
    packages = {}
    for member in manifest['workspace']['members']:
        package = tomllib.loads((root / member / 'Cargo.toml').read_text())
        packages[package['package']['name']] = package
    guide = root / 'docs/DEVELOPMENT.md'
    if guide.exists():
        for command in re.findall(r'`(cargo [^`]+)`', guide.read_text()):
            selected = re.findall(r'(?:-p|--package) ([\w-]+)', command)
            for name in selected:
                if name not in packages:
                    errors.append(f'DEVELOPMENT.md: unknown Cargo package {name}')
            for feature_list in re.findall(r'--features ([\w,/-]+)', command):
                for feature in feature_list.split(','):
                    names = selected
                    if '/' in feature:
                        name, feature = feature.split('/', 1)
                        names = [name]
                    for name in names:
                        if feature not in packages.get(name, {}).get('features', {}):
                            errors.append(f'DEVELOPMENT.md: {name} has no feature {feature}')
    bundle_dir = root / 'crates/basilica-cli/src/cli/handlers/skills'
    bundle_path = bundle_dir / 'bundle.json'
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text())
        if not re.fullmatch(r'[0-9a-f]{40}', bundle['revision']):
            errors.append('Skill bundle source must use a full immutable commit revision')
        if set(bundle['skills']) != set(bundle['files']):
            errors.append('Skill bundle file inventory must cover exactly the curated skills')
        with tarfile.open(bundle_dir / 'pinned-bundle.tar.gz') as archive:
            actual = {}
            for entry in archive:
                parts = Path(entry.name).parts
                if len(parts) < 4 or parts[1] != 'skills' or parts[2] not in bundle['skills']:
                    continue
                if entry.isfile():
                    data = archive.extractfile(entry).read()
                    actual.setdefault(parts[2], {})['/'.join(parts[3:])] = hashlib.sha256(data).hexdigest()
            if actual != bundle['files']:
                errors.append('Pinned skill archive differs from the manifest file hashes')
        for document in (root / '.claude/skills').glob('*/SKILL.md'):
            for revision in re.findall(r'basilica-skills/blob/([^/]+)/', document.read_text()):
                if revision != bundle['revision']:
                    errors.append(f'{document.relative_to(root)}: customer skill revision differs from CLI manifest')
    return errors


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    errors = check(root)
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        raise SystemExit(1)
    print('Agent instruction links, shared entry point and Cargo command declarations: OK')
