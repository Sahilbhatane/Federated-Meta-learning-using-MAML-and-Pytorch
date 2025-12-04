"""Run the repository's tests using pytest.

Usage:
  python -m test.run_all_tests
  or
  python test/run_all_tests.py

This script invokes pytest programmatically and returns a non-zero exit
code if any tests fail. It is intended as a simple single-entry test runner
for contributors and CI.
"""
import sys
import pytest


def main(argv=None):
    argv = argv or []
    # Run pytest discovery on the repository 'test' directory by default
    args = ["-q"] + argv
    # If no path provided, run against the test/ directory
    if not any(not a.startswith("-") for a in argv):
        args.append("test")
    return pytest.main(args)


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
