#!/usr/bin/env python3
"""
Test runner script for Stock Prediction Application.

This script runs pytest with coverage reporting for the envs and trainers modules.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run tests with coverage reporting."""
    print("🧪 Running Stock Prediction Application Tests")
    print("=" * 50)

    # Check if pytest is available
    try:
        import pytest
        import coverage
    except ImportError:
        print("❌ Test dependencies not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "pytest", "pytest-cov", "coverage"])

    # Run tests with coverage
    repo_root = Path(__file__).resolve().parent.parent
    tests_dir = (repo_root / "tests").as_posix()

    test_args = [
        "pytest",
        tests_dir,
        "-v",
        "--cov=envs",
        "--cov=trainers",
        "--cov=agent",
        "--cov=utils",
        "--cov=components",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=70",
        "--tb=short"
    ]

    print("📊 Running tests with coverage...")
    result = subprocess.run(test_args, cwd=repo_root)

    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print("📈 Coverage report generated in htmlcov/")
        print("📄 XML coverage report: coverage.xml")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


def run_specific_tests(module=None):
    """Run tests for specific modules."""

    repo_root = Path(__file__).resolve().parent.parent
    if module:
        test_path = (repo_root / "tests" / module).as_posix()
        print(f"🎯 Running tests for {module} module...")
    else:
        test_path = (repo_root / "tests").as_posix()
        print("🎯 Running all tests...")

    test_args = [
        "pytest",
        test_path,
        "-v",
        "--cov=envs",
        "--cov=trainers",
        "--cov=agent",
        "--cov=utils",
        "--cov=components",
        "--cov-report=term-missing",
        "--tb=short"
    ]

    result = subprocess.run(test_args, cwd=repo_root)
    return result.returncode == 0


def show_coverage_summary():
    """Show coverage summary."""
    try:
        result = subprocess.run([
            "coverage", "report", "--show-missing"
        ], cwd=Path(__file__).parent, capture_output=True, text=True)

        print("📊 Coverage Summary:")
        print(result.stdout)

        if result.stderr:
            print("Warnings:")
            print(result.stderr)

    except FileNotFoundError:
        print("❌ Coverage not available. Run tests first.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "envs":
            success = run_specific_tests("envs")
        elif command == "trainers":
            success = run_specific_tests("trainers")
        elif command == "coverage":
            show_coverage_summary()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: envs, trainers, coverage")
            sys.exit(1)
    else:
        run_tests()
