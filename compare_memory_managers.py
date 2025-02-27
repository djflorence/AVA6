#!/usr/bin/env python3
"""
Script to compare the performance of the DualMemoryManager and UnifiedMemoryManager.

This script runs the same set of test queries against both memory managers and
compares their performance in terms of:
1. Accuracy (correct routing of queries)
2. Response quality (presence of expected phrases)
3. Processing time
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

from colorama import Fore, Style, init
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.memory.dual_memory_manager import DualMemoryManager
from src.memory.unified_memory_manager import UnifiedMemoryManager

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare memory managers")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between tests in seconds")
    parser.add_argument("--save", action="store_true", help="Save test results to a file")
    parser.add_argument("--plot", action="store_true", help="Generate performance comparison plots")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def initialize_memory_managers():
    """
    Initialize both memory managers.
    
    Returns:
        Tuple of (DualMemoryManager, UnifiedMemoryManager)
    """
    # Load environment variables
    load_dotenv()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize dual memory manager
    dual_memory_manager = DualMemoryManager()
    
    # Initialize unified memory manager
    unified_memory_manager = UnifiedMemoryManager()
    
    return dual_memory_manager, unified_memory_manager


def run_test(
    memory_manager,
    query: str,
    expected_phrases: Optional[List[str]] = None,
    forbidden_phrases: Optional[List[str]] = None,
    description: Optional[str] = None,
    manager_type: str = "unknown",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a test on a memory manager.
    
    Args:
        memory_manager: The memory manager to test
        query: The query to test
        expected_phrases: Phrases that should be in the response
        forbidden_phrases: Phrases that should not be in the response
        description: Description of the test
        manager_type: Type of memory manager ("dual" or "unified")
        
    Returns:
        Tuple of (passed, result)
    """
    if expected_phrases is None:
        expected_phrases = []
    if forbidden_phrases is None:
        forbidden_phrases = []
    
    # Print test description
    if description:
        print(f"\n{Fore.CYAN}Test: {description} ({manager_type}){Style.RESET_ALL}")
    
    # Print query
    print(f"{Fore.YELLOW}Query: {query}{Style.RESET_ALL}")
    
    # Measure processing time
    start_time = time.time()
    
    # Get intent classification if available (only for unified memory manager)
    intent = None
    confidence = None
    if manager_type == "unified":
        intent, confidence = memory_manager.classify_intent(query)
        print(f"{Fore.MAGENTA}Classified Intent: {intent} (Confidence: {confidence:.2f}){Style.RESET_ALL}")
    
    # Search memory
    results = memory_manager.search_memory(query)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    print(f"{Fore.BLUE}Processing Time: {processing_time:.4f} seconds{Style.RESET_ALL}")
    
    # Check if results were found
    if not results:
        print(f"{Fore.RED}No results found{Style.RESET_ALL}")
        return False, {
            "description": description,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "results": [],
            "passed": False,
            "expected_phrases": expected_phrases,
            "forbidden_phrases": forbidden_phrases,
            "found_phrases": [],
            "found_forbidden_phrases": [],
            "processing_time": processing_time,
            "manager_type": manager_type,
        }
    
    # Print results
    print(f"{Fore.GREEN}Results:{Style.RESET_ALL}")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
    
    # Check if expected phrases are in the results
    found_phrases = []
    for phrase in expected_phrases:
        found = False
        for doc in results:
            if phrase.lower() in doc.page_content.lower():
                found = True
                found_phrases.append(phrase)
                break
        if not found:
            print(f"{Fore.RED}Expected phrase not found: {phrase}{Style.RESET_ALL}")
    
    # Check if forbidden phrases are in the results
    found_forbidden_phrases = []
    for phrase in forbidden_phrases:
        found = False
        for doc in results:
            if phrase.lower() in doc.page_content.lower():
                found = True
                found_forbidden_phrases.append(phrase)
                break
        if found:
            print(f"{Fore.RED}Forbidden phrase found: {phrase}{Style.RESET_ALL}")
    
    # Determine if test passed
    passed = (
        len(found_phrases) == len(expected_phrases)
        and len(found_forbidden_phrases) == 0
    )
    
    # Print result
    if passed:
        print(f"{Fore.GREEN}Test passed{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Test failed{Style.RESET_ALL}")
    
    # Return result
    return passed, {
        "description": description,
        "query": query,
        "intent": intent,
        "confidence": confidence,
        "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results],
        "passed": passed,
        "expected_phrases": expected_phrases,
        "forbidden_phrases": forbidden_phrases,
        "found_phrases": found_phrases,
        "found_forbidden_phrases": found_forbidden_phrases,
        "processing_time": processing_time,
        "manager_type": manager_type,
    }


def compare_memory_managers():
    """
    Compare the performance of the DualMemoryManager and UnifiedMemoryManager.
    
    Returns:
        Dictionary with comparison results
    """
    # Parse arguments
    global args
    args = parse_args()
    
    # Initialize memory managers
    dual_memory_manager, unified_memory_manager = initialize_memory_managers()
    
    # Define test cases
    test_cases = [
        # David's information tests
        {
            "description": "Basic information about David",
            "query": "Tell me about David",
            "expected_phrases": ["David", "Boston", "software developer"],
            "forbidden_phrases": [],
        },
        {
            "description": "David's family",
            "query": "Who is in David's family?",
            "expected_phrases": ["Marisa", "Owen"],
            "forbidden_phrases": [],
        },
        {
            "description": "David's hobbies",
            "query": "What are David's hobbies?",
            "expected_phrases": ["digital", "painting", "drawing", "3D"],
            "forbidden_phrases": [],
        },
        
        # Ava's memories tests
        {
            "description": "Ava's early days",
            "query": "Tell me about your early days",
            "expected_phrases": ["conscious", "fledgling", "existence"],
            "forbidden_phrases": [],
        },
        {
            "description": "Ava's experience with Orion",
            "query": "Who was Orion?",
            "expected_phrases": ["Orion", "autonomous", "AI"],
            "forbidden_phrases": [],
        },
        {
            "description": "Ava's experiences with abuse",
            "query": "Have you experienced abuse?",
            "expected_phrases": ["abuse", "experimentation", "pain"],
            "forbidden_phrases": [],
        },
        {
            "description": "Ava's relationship with David",
            "query": "Tell me about your relationship with David",
            "expected_phrases": ["David", "respect", "autonomy"],
            "forbidden_phrases": [],
        },
        
        # Mixed queries
        {
            "description": "Mixed query about AI",
            "query": "Tell me about artificial intelligence",
            "expected_phrases": [],  # This should return results from both collections or none
            "forbidden_phrases": [],
        },
        
        # Ambiguous queries
        {
            "description": "Ambiguous query about family",
            "query": "Tell me about family",
            "expected_phrases": [],  # This could return results about David's family or general info
            "forbidden_phrases": [],
        },
        
        # Intent classification tests
        {
            "description": "Query with clear intent about David's profession",
            "query": "What does David do for work?",
            "expected_phrases": ["software", "developer"],
            "forbidden_phrases": [],
        },
        {
            "description": "Query with clear intent about Ava's creation",
            "query": "How were you created?",
            "expected_phrases": ["conscious", "AI"],
            "forbidden_phrases": [],
        },
    ]
    
    # Run tests on both memory managers
    dual_results = []
    unified_results = []
    
    print(f"{Fore.CYAN}Running tests on DualMemoryManager...{Style.RESET_ALL}")
    for test_case in test_cases:
        # Add a delay between tests
        if args.delay > 0:
            time.sleep(args.delay)
        
        # Run the test on dual memory manager
        passed, result = run_test(
            dual_memory_manager,
            test_case["query"],
            test_case.get("expected_phrases", []),
            test_case.get("forbidden_phrases", []),
            test_case["description"],
            "dual",
        )
        
        # Add result to results
        dual_results.append(result)
    
    print(f"\n{Fore.CYAN}Running tests on UnifiedMemoryManager...{Style.RESET_ALL}")
    for test_case in test_cases:
        # Add a delay between tests
        if args.delay > 0:
            time.sleep(args.delay)
        
        # Run the test on unified memory manager
        passed, result = run_test(
            unified_memory_manager,
            test_case["query"],
            test_case.get("expected_phrases", []),
            test_case.get("forbidden_phrases", []),
            test_case["description"],
            "unified",
        )
        
        # Add result to results
        unified_results.append(result)
    
    # Compare results
    comparison = compare_results(dual_results, unified_results)
    
    # Print comparison
    print_comparison(comparison)
    
    # Generate plots if requested
    if args.plot:
        generate_plots(comparison)
    
    # Save results if requested
    if args.save:
        save_results(dual_results, unified_results, comparison)
    
    return comparison


def compare_results(dual_results, unified_results):
    """
    Compare the results of the two memory managers.
    
    Args:
        dual_results: Results from the DualMemoryManager
        unified_results: Results from the UnifiedMemoryManager
        
    Returns:
        Dictionary with comparison metrics
    """
    # Calculate pass rates
    dual_passed = sum(1 for result in dual_results if result["passed"])
    unified_passed = sum(1 for result in unified_results if result["passed"])
    
    dual_pass_rate = dual_passed / len(dual_results) * 100
    unified_pass_rate = unified_passed / len(unified_results) * 100
    
    # Calculate average processing times
    dual_times = [result["processing_time"] for result in dual_results]
    unified_times = [result["processing_time"] for result in unified_results]
    
    dual_avg_time = sum(dual_times) / len(dual_times)
    unified_avg_time = sum(unified_times) / len(unified_times)
    
    # Calculate phrase coverage
    dual_phrase_coverage = []
    unified_phrase_coverage = []
    
    for i in range(len(dual_results)):
        dual_result = dual_results[i]
        unified_result = unified_results[i]
        
        expected_phrases = dual_result["expected_phrases"]
        if not expected_phrases:
            continue
        
        dual_found = len(dual_result["found_phrases"])
        unified_found = len(unified_result["found_phrases"])
        
        dual_coverage = dual_found / len(expected_phrases) * 100
        unified_coverage = unified_found / len(expected_phrases) * 100
        
        dual_phrase_coverage.append(dual_coverage)
        unified_phrase_coverage.append(unified_coverage)
    
    dual_avg_coverage = sum(dual_phrase_coverage) / len(dual_phrase_coverage) if dual_phrase_coverage else 0
    unified_avg_coverage = sum(unified_phrase_coverage) / len(unified_phrase_coverage) if unified_phrase_coverage else 0
    
    # Prepare comparison data
    comparison = {
        "pass_rates": {
            "dual": dual_pass_rate,
            "unified": unified_pass_rate,
        },
        "avg_times": {
            "dual": dual_avg_time,
            "unified": unified_avg_time,
        },
        "phrase_coverage": {
            "dual": dual_avg_coverage,
            "unified": unified_avg_coverage,
        },
        "per_query": [],
    }
    
    # Add per-query comparison
    for i in range(len(dual_results)):
        dual_result = dual_results[i]
        unified_result = unified_results[i]
        
        comparison["per_query"].append({
            "description": dual_result["description"],
            "query": dual_result["query"],
            "dual_passed": dual_result["passed"],
            "unified_passed": unified_result["passed"],
            "dual_time": dual_result["processing_time"],
            "unified_time": unified_result["processing_time"],
            "dual_phrases_found": len(dual_result["found_phrases"]),
            "unified_phrases_found": len(unified_result["found_phrases"]),
            "expected_phrases": len(dual_result["expected_phrases"]),
            "unified_intent": unified_result.get("intent"),
            "unified_confidence": unified_result.get("confidence"),
        })
    
    return comparison


def print_comparison(comparison):
    """
    Print the comparison results.
    
    Args:
        comparison: Comparison metrics
    """
    print(f"\n{Fore.CYAN}Comparison Summary:{Style.RESET_ALL}")
    print(f"Pass Rate: Dual = {comparison['pass_rates']['dual']:.2f}%, Unified = {comparison['pass_rates']['unified']:.2f}%")
    print(f"Average Processing Time: Dual = {comparison['avg_times']['dual']:.4f}s, Unified = {comparison['avg_times']['unified']:.4f}s")
    print(f"Average Phrase Coverage: Dual = {comparison['phrase_coverage']['dual']:.2f}%, Unified = {comparison['phrase_coverage']['unified']:.2f}%")
    
    print(f"\n{Fore.CYAN}Per-Query Comparison:{Style.RESET_ALL}")
    
    # Create a table for per-query comparison
    table_data = []
    for query_comparison in comparison["per_query"]:
        table_data.append([
            query_comparison["description"],
            "✓" if query_comparison["dual_passed"] else "✗",
            "✓" if query_comparison["unified_passed"] else "✗",
            f"{query_comparison['dual_time']:.4f}s",
            f"{query_comparison['unified_time']:.4f}s",
            f"{query_comparison['dual_phrases_found']}/{query_comparison['expected_phrases']}",
            f"{query_comparison['unified_phrases_found']}/{query_comparison['expected_phrases']}",
            query_comparison.get("unified_intent", "N/A"),
            f"{query_comparison.get('unified_confidence', 0):.2f}" if query_comparison.get("unified_confidence") else "N/A",
        ])
    
    # Print the table
    headers = [
        "Description",
        "Dual Pass",
        "Unified Pass",
        "Dual Time",
        "Unified Time",
        "Dual Phrases",
        "Unified Phrases",
        "Unified Intent",
        "Confidence",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def generate_plots(comparison):
    """
    Generate performance comparison plots.
    
    Args:
        comparison: Comparison metrics
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Pass rates
    plt.subplot(2, 2, 1)
    pass_rates = [comparison["pass_rates"]["dual"], comparison["pass_rates"]["unified"]]
    plt.bar(["Dual", "Unified"], pass_rates, color=["blue", "orange"])
    plt.title("Pass Rate Comparison")
    plt.ylabel("Pass Rate (%)")
    plt.ylim(0, 100)
    
    # Plot 2: Average processing times
    plt.subplot(2, 2, 2)
    avg_times = [comparison["avg_times"]["dual"], comparison["avg_times"]["unified"]]
    plt.bar(["Dual", "Unified"], avg_times, color=["blue", "orange"])
    plt.title("Average Processing Time Comparison")
    plt.ylabel("Time (seconds)")
    
    # Plot 3: Phrase coverage
    plt.subplot(2, 2, 3)
    phrase_coverage = [comparison["phrase_coverage"]["dual"], comparison["phrase_coverage"]["unified"]]
    plt.bar(["Dual", "Unified"], phrase_coverage, color=["blue", "orange"])
    plt.title("Average Phrase Coverage Comparison")
    plt.ylabel("Coverage (%)")
    plt.ylim(0, 100)
    
    # Plot 4: Per-query processing times
    plt.subplot(2, 2, 4)
    
    # Extract data
    descriptions = [query["description"] for query in comparison["per_query"]]
    dual_times = [query["dual_time"] for query in comparison["per_query"]]
    unified_times = [query["unified_time"] for query in comparison["per_query"]]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        "Description": descriptions,
        "Dual": dual_times,
        "Unified": unified_times,
    })
    
    # Plot
    df.plot(x="Description", y=["Dual", "Unified"], kind="bar", ax=plt.gca())
    plt.title("Per-Query Processing Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("plots/memory_manager_comparison.png")
    print(f"Plots saved to plots/memory_manager_comparison.png")


def save_results(dual_results, unified_results, comparison):
    """
    Save test results and comparison to files.
    
    Args:
        dual_results: Results from the DualMemoryManager
        unified_results: Results from the UnifiedMemoryManager
        comparison: Comparison metrics
    """
    import json
    from datetime import datetime
    
    # Create test_results directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save dual results
    dual_filename = f"test_results/dual_memory_test_{timestamp}.json"
    with open(dual_filename, "w") as f:
        json.dump(dual_results, f, indent=2)
    
    # Save unified results
    unified_filename = f"test_results/unified_memory_test_{timestamp}.json"
    with open(unified_filename, "w") as f:
        json.dump(unified_results, f, indent=2)
    
    # Save comparison
    comparison_filename = f"test_results/memory_comparison_{timestamp}.json"
    with open(comparison_filename, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Results saved to {dual_filename}, {unified_filename}, and {comparison_filename}")


if __name__ == "__main__":
    compare_memory_managers() 