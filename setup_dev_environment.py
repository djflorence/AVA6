#!/usr/bin/env python3
"""
Set up the development environment for the Ava project.

This script:
1. Installs required dependencies
2. Sets up pre-commit hooks
3. Configures the development environment

Usage:
    python setup_dev_environment.py [--skip-deps] [--skip-hooks]

Options:
    --skip-deps     Skip installing dependencies
    --skip-hooks    Skip setting up pre-commit hooks
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
logger = logging.getLogger("ava_setup")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up the development environment")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip installing dependencies"
    )
    parser.add_argument(
        "--skip-hooks", action="store_true", help="Skip setting up pre-commit hooks"
    )
    return parser.parse_args()


def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")

    try:
        # Install regular dependencies
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
        )

        # Install pre-commit
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pre-commit"],
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False


def setup_pre_commit_hooks():
    """Set up pre-commit hooks."""
    logger.info("Setting up pre-commit hooks...")

    try:
        # Check if .pre-commit-config.yaml exists
        if not Path(".pre-commit-config.yaml").exists():
            logger.error(".pre-commit-config.yaml not found")
            return False

        # Install pre-commit hooks
        subprocess.run(
            ["pre-commit", "install"], check=True, capture_output=True, text=True
        )

        # Run pre-commit hooks on all files
        logger.info("Running pre-commit hooks on all files (this may take a while)...")
        subprocess.run(
            ["pre-commit", "run", "--all-files"],
            check=False,  # Don't fail if hooks find issues
            capture_output=False,
        )

        logger.info("Pre-commit hooks set up successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up pre-commit hooks: {e}")
        return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            sys.exit(1)

    # Set up pre-commit hooks
    if not args.skip_hooks:
        if not setup_pre_commit_hooks():
            logger.error("Failed to set up pre-commit hooks")
            sys.exit(1)

    logger.info("\nDevelopment environment set up successfully!")
    logger.info("\nYou can now:")
    logger.info("  1. Run the linting checks with: python lint.py --check")
    logger.info("  2. Fix linting issues with: python lint.py --fix")
    logger.info("  3. Run the tests with: python test_ava_chat.py")


if __name__ == "__main__":
    main()
