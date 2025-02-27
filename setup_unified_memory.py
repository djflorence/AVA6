#!/usr/bin/env python3
"""
Setup script for the Unified Memory System.

This script:
1. Prepares the unified memory collection
2. Tests the unified memory manager
3. Compares the unified memory manager with the dual memory manager
4. Generates reports and visualizations
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("unified_memory_setup.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up and test the Unified Memory System")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip memory preparation step")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing step")
    parser.add_argument("--skip-compare", action="store_true", help="Skip comparison step")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def run_command(command, description):
    """
    Run a command and log the output.
    
    Args:
        command: The command to run
        description: Description of the command
        
    Returns:
        True if the command succeeded, False otherwise
    """
    logger.info(f"Running {description}...")
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.info(f"{description} completed successfully")
        logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with error code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False


def prepare_unified_memory():
    """Prepare the unified memory collection."""
    return run_command(
        ["python", "src/memory/prepare_unified_memory.py"],
        "Unified memory preparation",
    )


def test_unified_memory():
    """Test the unified memory manager."""
    return run_command(
        ["python", "test_unified_memory.py", "--save"],
        "Unified memory testing",
    )


def compare_memory_managers():
    """Compare the unified memory manager with the dual memory manager."""
    return run_command(
        ["python", "compare_memory_managers.py", "--save", "--plot"],
        "Memory manager comparison",
    )


def create_summary_report():
    """Create a summary report of the setup and testing process."""
    logger.info("Creating summary report...")
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create report file
    report_filename = f"reports/unified_memory_summary_{timestamp}.md"
    
    # Find the latest test results
    test_results_dir = "test_results"
    if os.path.exists(test_results_dir):
        test_files = [f for f in os.listdir(test_results_dir) if f.startswith("unified_memory_test_")]
        test_files.sort(reverse=True)
        latest_test_file = test_files[0] if test_files else None
        
        comparison_files = [f for f in os.listdir(test_results_dir) if f.startswith("memory_comparison_")]
        comparison_files.sort(reverse=True)
        latest_comparison_file = comparison_files[0] if comparison_files else None
    else:
        latest_test_file = None
        latest_comparison_file = None
    
    # Write report
    with open(report_filename, "w") as f:
        f.write("# Unified Memory System Setup Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Setup Process\n\n")
        f.write("1. **Memory Preparation**: ")
        if args.skip_prepare:
            f.write("Skipped\n")
        else:
            f.write("Completed\n")
        
        f.write("2. **Memory Testing**: ")
        if args.skip_test:
            f.write("Skipped\n")
        else:
            f.write("Completed\n")
        
        f.write("3. **Memory Comparison**: ")
        if args.skip_compare:
            f.write("Skipped\n")
        else:
            f.write("Completed\n")
        
        f.write("\n## Test Results\n\n")
        if latest_test_file:
            f.write(f"Latest test results: `{latest_test_file}`\n\n")
            f.write("See the test results file for detailed information.\n\n")
        else:
            f.write("No test results found.\n\n")
        
        f.write("\n## Comparison Results\n\n")
        if latest_comparison_file:
            f.write(f"Latest comparison results: `{latest_comparison_file}`\n\n")
            f.write("See the comparison results file for detailed information.\n\n")
            
            # Add plot if available
            if os.path.exists("plots/memory_manager_comparison.png"):
                f.write("### Performance Comparison\n\n")
                f.write("![Memory Manager Comparison](../plots/memory_manager_comparison.png)\n\n")
        else:
            f.write("No comparison results found.\n\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review the test and comparison results\n")
        f.write("2. Make any necessary adjustments to the unified memory system\n")
        f.write("3. Consider implementing the future improvements outlined in the README\n")
    
    logger.info(f"Summary report created: {report_filename}")
    return report_filename


def main():
    """Main function."""
    global args
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Unified Memory System setup...")
    
    # Record start time
    start_time = time.time()
    
    # Prepare unified memory
    if not args.skip_prepare:
        if not prepare_unified_memory():
            logger.error("Memory preparation failed. Exiting.")
            return 1
    else:
        logger.info("Skipping memory preparation step")
    
    # Test unified memory
    if not args.skip_test:
        if not test_unified_memory():
            logger.warning("Memory testing failed. Continuing with setup.")
    else:
        logger.info("Skipping memory testing step")
    
    # Compare memory managers
    if not args.skip_compare:
        if not compare_memory_managers():
            logger.warning("Memory comparison failed. Continuing with setup.")
    else:
        logger.info("Skipping memory comparison step")
    
    # Create summary report
    report_filename = create_summary_report()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Unified Memory System setup completed in {elapsed_time:.2f} seconds")
    logger.info(f"Summary report: {report_filename}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 