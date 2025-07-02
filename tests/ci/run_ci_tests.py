#!/usr/bin/env python
"""Run CI-specific tests with appropriate configuration."""

import sys
import subprocess
from pathlib import Path


def run_ci_tests():
    """Run all CI tests with proper markers and configuration."""

    # Get the CI test directory
    ci_test_dir = Path(__file__).parent

    # Pytest arguments for CI
    pytest_args = [
        "pytest",
        str(ci_test_dir),  # Only run tests in CI directory
        "-v",  # Verbose output
        "-m",
        "ci",  # Only run tests marked as CI
        "--tb=short",  # Short traceback format
        "--no-header",  # No pytest header
        "-p",
        "no:warnings",  # Disable warnings
        "--maxfail=5",  # Stop after 5 failures
        "--durations=10",  # Show 10 slowest tests
    ]

    # Add coverage if requested
    if "--coverage" in sys.argv:
        pytest_args.extend(
            [
                "--cov=training_lens",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_ci",
                "--cov-fail-under=50",  # Lower threshold for CI tests
            ]
        )

    # Add specific test file if provided
    if len(sys.argv) > 1 and sys.argv[1] != "--coverage":
        test_file = sys.argv[1]
        if not test_file.startswith("test_"):
            test_file = f"test_{test_file}"
        if not test_file.endswith(".py"):
            test_file = f"{test_file}.py"
        pytest_args[1] = str(ci_test_dir / test_file)

    print(f"Running CI tests with command: {' '.join(pytest_args)}")
    print("-" * 80)

    # Run pytest
    result = subprocess.run(pytest_args)

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_ci_tests())
