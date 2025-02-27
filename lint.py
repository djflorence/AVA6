#!/usr/bin/env python3
"""
Lint helper for the Ava project.

This script helps maintain code quality by running linters and formatters
on the codebase. It can check for issues or automatically fix them.

Usage:
    python lint.py [--check] [--fix] [--path PATH]

Options:
    --check     Check for linting issues without fixing them
    --fix       Automatically fix linting issues (default)
    --path PATH Path to check/fix (default: src)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ava_lint")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Lint helper for the Ava project")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for linting issues without fixing them",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix linting issues"
    )
    parser.add_argument("--path", type=str, default="src", help="Path to check/fix")
    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are installed."""
    required_tools = ["flake8", "black", "isort"]
    missing_tools = []

    for tool in required_tools:
        try:
            subprocess.run(
                [tool, "--version"], check=True, capture_output=True, text=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)

    if missing_tools:
        logger.error(f"Missing required tools: {', '.join(missing_tools)}")
        logger.error("Please install them with: pip install flake8 black isort")
        return False

    return True


def run_flake8(path, fix=False):
    """
    Run flake8 to check for linting issues.

    Args:
        path: Path to check
        fix: Whether to fix issues (flake8 doesn't fix, just reports)

    Returns:
        True if no issues found, False otherwise
    """
    logger.info(f"Running flake8 on {path}...")

    try:
        result = subprocess.run(
            ["flake8", path], check=False, capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.warning("Flake8 found issues:")
            for line in result.stdout.splitlines():
                logger.warning(f"  {line}")
            return False

        logger.info("Flake8 check passed!")
        return True
    except Exception as e:
        logger.error(f"Error running flake8: {str(e)}")
        return False


def run_black(path, fix=False):
    """
    Run black to check/fix code formatting.

    Args:
        path: Path to check/fix
        fix: Whether to fix issues

    Returns:
        True if no issues found or fixed, False otherwise
    """
    logger.info(f"Running black on {path}...")

    cmd = ["black"]
    if not fix:
        cmd.append("--check")
    cmd.append(path)

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if result.returncode != 0:
            if fix:
                logger.info("Black fixed formatting issues")
            else:
                logger.warning("Black found formatting issues:")
                for line in result.stderr.splitlines():
                    if line:
                        logger.warning(f"  {line}")
            return False

        logger.info("Black check passed!")
        return True
    except Exception as e:
        logger.error(f"Error running black: {str(e)}")
        return False


def run_isort(path, fix=False):
    """
    Run isort to check/fix import ordering.

    Args:
        path: Path to check/fix
        fix: Whether to fix issues

    Returns:
        True if no issues found or fixed, False otherwise
    """
    logger.info(f"Running isort on {path}...")

    cmd = ["isort"]
    if not fix:
        cmd.append("--check")
    cmd.append(path)

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if result.returncode != 0:
            if fix:
                logger.info("isort fixed import ordering issues")
            else:
                logger.warning("isort found import ordering issues:")
                for line in result.stdout.splitlines():
                    if line:
                        logger.warning(f"  {line}")
            return False

        logger.info("isort check passed!")
        return True
    except Exception as e:
        logger.error(f"Error running isort: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Default to fix if neither check nor fix is specified
    if not args.check and not args.fix:
        args.fix = True

    # Check if required dependencies are installed
    if not check_dependencies():
        sys.exit(1)

    # Check if path exists
    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    # Run linters
    flake8_result = run_flake8(args.path, fix=False)  # flake8 doesn't fix
    black_result = run_black(args.path, fix=args.fix)
    isort_result = run_isort(args.path, fix=args.fix)

    # Print summary
    logger.info("\nLinting Summary:")
    logger.info(f"  flake8: {'✅ Passed' if flake8_result else '❌ Failed'}")
    logger.info(
        f"  black: {'✅ Passed' if black_result else '✅ Fixed' if args.fix else '❌ Failed'}"
    )
    logger.info(
        f"  isort: {'✅ Passed' if isort_result else '✅ Fixed' if args.fix else '❌ Failed'}"
    )

    # Exit with error if any linter failed
    if args.check and not all([flake8_result, black_result, isort_result]):
        logger.error("\nLinting failed! Run with --fix to automatically fix issues.")
        sys.exit(1)

    logger.info("\nLinting completed successfully!")


if __name__ == "__main__":
    main()
