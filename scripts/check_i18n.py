#!/usr/bin/env python3
"""Check that all i18n translation files have matching keys.

This script ensures that en.json and es.json (and any future language files)
have the same set of translation keys. Missing translations are reported
as errors to prevent incomplete localizations from being committed.
"""

import json
import sys
from pathlib import Path


def flatten_keys(data: dict, prefix: str = "") -> set[str]:
    """Recursively flatten nested dict keys into dot-notation strings."""
    keys = set()
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.update(flatten_keys(value, full_key))
        else:
            keys.add(full_key)
    return keys


def check_i18n_files() -> int:
    """Check all i18n files for key completeness. Returns exit code."""
    i18n_dir = Path(__file__).parent.parent / "frontend" / "src" / "i18n"

    if not i18n_dir.exists():
        print(f"i18n directory not found: {i18n_dir}")
        return 1

    # Find all JSON files
    json_files = list(i18n_dir.glob("*.json"))

    if len(json_files) < 2:
        print("Need at least 2 language files to compare")
        return 0  # Not an error, just nothing to compare

    # Load all files and extract keys
    file_keys: dict[str, set[str]] = {}
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            file_keys[json_file.name] = flatten_keys(data)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {json_file.name}: {e}")
            return 1

    # Use the first file (alphabetically) as the reference
    reference_file = sorted(file_keys.keys())[0]
    reference_keys = file_keys[reference_file]

    errors_found = False

    # Compare each file against the reference
    for filename, keys in sorted(file_keys.items()):
        if filename == reference_file:
            continue

        missing_in_target = reference_keys - keys
        extra_in_target = keys - reference_keys

        if missing_in_target:
            errors_found = True
            print(f"\n{filename} is missing {len(missing_in_target)} key(s) from {reference_file}:")
            for key in sorted(missing_in_target):
                print(f"  - {key}")

        if extra_in_target:
            errors_found = True
            print(f"\n{filename} has {len(extra_in_target)} extra key(s) not in {reference_file}:")
            for key in sorted(extra_in_target):
                print(f"  + {key}")

    if errors_found:
        print("\ni18n check failed: translation keys are out of sync")
        return 1

    print(f"i18n check passed: {len(json_files)} files with {len(reference_keys)} keys each")
    return 0


if __name__ == "__main__":
    sys.exit(check_i18n_files())
