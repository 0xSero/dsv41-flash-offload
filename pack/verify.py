#!/usr/bin/env python3
"""Verify the actual overlay files and generated tensors, not just receipt claims."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open('rb') as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def verify(pack, want):
    errors = []
    def check(name, expected):
        if not isinstance(name, str) or Path(name).name != name:
            errors.append(f'invalid generated file: {name}')
            return
        try:
            actual = sha256(pack / name)
        except OSError:
            errors.append(f'missing generated file: {name}')
            return
        if actual != expected:
            errors.append(f'hash mismatch: {name}')
    for name, expected in want['final_files'].items():
        check(name, expected)
    index = json.loads((pack / 'model.safetensors.index.json').read_text())['weight_map']
    receipt = json.loads((pack / 'adapter-receipt.json').read_text())['grouped_woa']
    keys = [r['key'] for r in receipt]
    if set(keys) != set(want['grouped_woa']) or len(keys) != len(set(keys)):
        errors.append('grouped wo_a receipt is incomplete or duplicated')
    for key, expected in want['grouped_woa'].items():
        check(index.get(key), expected)
    dense = json.loads((pack / 'native-dense-receipt.json').read_text())
    check('native-dense.safetensors', dense.get('sha256'))
    for name in ('Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw', 'DeepSeek-V4.1-Flash-engram'):
        if not (pack.parent / name).is_dir():
            errors.append(f'missing sibling input directory: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=Path('/models/DSV41-EXL3-3090-D010'))
    args = parser.parse_args()
    want = json.loads((Path(__file__).parent / 'expected-sha256.json').read_text())
    try:
        errors = verify(args.pack, want)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors = [f'incomplete pack: {error}']
    for error in errors:
        print('FAIL', error)
    if not errors:
        print('OK overlay metadata, grouped wo_a files and native dense output')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
