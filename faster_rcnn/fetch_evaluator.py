"""Acquire the pinned evaluator wheel without installing it or altering metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / 'evaluator.lock.json'


def verify_wheel(path, lock=None):
    lock = lock or json.loads(LOCK.read_text(encoding='utf-8'))
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != lock['sha256']:
        raise ValueError('Evaluator wheel SHA256 mismatch; refusing to load it')
    return actual


def acquire(destination=None):
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    destination = Path(destination) if destination else ROOT / 'vendor' / lock['filename']
    if destination.exists():
        verify_wheel(destination, lock)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A temporary sibling allows verification before the final file is published.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.download', delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(lock['url'], timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        verify_wheel(temporary, lock)
        if destination.exists():
            verify_wheel(destination, lock)
        else:
            temporary.replace(destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the existing wheel without network access')
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if args.check:
        verify_wheel(ROOT / 'vendor' / lock['filename'], lock)
    else:
        acquire()
    print('Pinned Ultralytics evaluator SHA256 verified')


if __name__ == '__main__':
    main()
