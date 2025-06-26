#!/usr/bin/env python3
"""
Comprehensive test script for the RAG service.
This script provides various testing options to validate the RAG service functionality.
"""

import requests
import json
import time
import argparse
import sys
from typing import List, Dict, Any, Optional

def send_request(url: str, queries: List[str], topk: int = 3, return_scores: bool = True) -> Dict[str, Any]:
    """
    Send a request to the RAG service and return the response.
    
    Args:
        url: The URL of the RAG service endpoint
        queries: List of query strings
        topk: Number of documents to retrieve per query
        return_scores: Whether to return relevance scores
        
    Returns:
        The JSON response from the RAG service
    """
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": return_scores
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def print_results(result: Dict[str, Any], queries: List[str], return_scores: bool) -> None:
    """Print the results in a readable format."""
    for i, query_result in enumerate(result["result"]):
        print(f"\nQuery {i+1}: '{queries[i]}'")
        
        if return_scores:
            # Results with scores
            for j, item in enumerate(query_result):
                doc = item["document"]
                score = item["score"]
                
                title = doc.get("title", "No title")
                text = doc.get("text", doc.get("contents", "No content"))
                
                # Truncate text for display
                if len(text) > 200:
                    text = text[:200] + "..."
                
                print(f"  Result {j+1} (Score: {score:.4f}):")
                print(f"    Title: {title}")
                print(f"    Text: {text}")
        else:
            # Results without scores
            for j, doc in enumerate(query_result):
                title = doc.get("title", "No title")
                text = doc.get("text", doc.get("contents", "No content"))
                
                if len(text) > 200:
                    text = text[:200] + "..."
                
                print(f"  Result {j+1}:")
                print(f"    Title: {title}")
                print(f"    Text: {text}")

def basic_connectivity_test(url: str) -> bool:
    """Test basic connectivity to the RAG service."""
    print("\n===== Basic Connectivity Test =====")
    try:
        queries = ["Test query"]
        result = send_request(url, queries)
        print("✅ Connection successful!")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def parameter_test(url: str) -> bool:
    """Test different parameter combinations."""
    print("\n===== Parameter Test =====")
    test_cases = [
        {"queries": ["What is machine learning?"], "topk": 1, "return_scores": True},
        {"queries": ["What is machine learning?"], "topk": 5, "return_scores": True},
        {"queries": ["What is machine learning?"], "topk": 3, "return_scores": False}
    ]
    
    success = True
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  Queries: {test_case['queries']}")
        print(f"  Top-k: {test_case['topk']}")
        print(f"  Return scores: {test_case['return_scores']}")
        
        try:
            result = send_request(
                url, 
                test_case["queries"], 
                test_case["topk"], 
                test_case["return_scores"]
            )
            print("  ✅ Request successful")
            print_results(result, test_case["queries"], test_case["return_scores"])
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            success = False
    
    return success

def performance_test(url: str, num_queries: int = 5) -> bool:
    """Test performance with multiple queries."""
    print("\n===== Performance Test =====")
    queries = [
        "What is machine learning?",
        "Explain reinforcement learning",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain natural language processing",
        "What is computer vision?",
        "How does transfer learning work?",
        "What is unsupervised learning?",
        "Explain gradient descent",
        "What is backpropagation?"
    ]
    
    # Use only the specified number of queries
    test_queries = queries[:num_queries]
    
    try:
        start_time = time.time()
        result = send_request(url, test_queries)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"✅ Successfully processed {num_queries} queries in {elapsed_time:.2f} seconds")
        print(f"  Average time per query: {elapsed_time/num_queries:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def error_handling_test(url: str) -> bool:
    """Test error handling with invalid requests."""
    print("\n===== Error Handling Test =====")
    test_cases = [
        {"name": "Empty query list", "payload": {"queries": [], "topk": 3}},
        {"name": "Invalid topk value", "payload": {"queries": ["Test"], "topk": -1}},
        {"name": "Very large topk", "payload": {"queries": ["Test"], "topk": 1000}}
    ]
    
    success = True
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            response = requests.post(url, json=test_case["payload"])
            if response.status_code >= 400:
                print(f"  ✅ Expected error response received: {response.status_code}")
                print(f"  Error message: {response.text}")
            else:
                print(f"  ⚠️ Warning: Expected error but got success response: {response.status_code}")
                success = False
        except Exception as e:
            print(f"  ✅ Expected exception: {e}")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Test the RAG service")
    parser.add_argument("--url", default="http://localhost:5004/retrieve", help="URL of the RAG service")
    parser.add_argument("--test", choices=["basic", "params", "perf", "error", "all"], default="all", 
                        help="Test type to run")
    parser.add_argument("--queries", nargs="+", help="Custom queries to test")
    parser.add_argument("--topk", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--no-scores", action="store_true", help="Don't return scores")
    
    args = parser.parse_args()
    
    print(f"Testing RAG service at: {args.url}")
    
    # Run the selected test
    if args.test == "basic" or args.test == "all":
        basic_connectivity_test(args.url)
    
    if args.test == "params" or args.test == "all":
        parameter_test(args.url)
    
    if args.test == "perf" or args.test == "all":
        performance_test(args.url)
    
    if args.test == "error" or args.test == "all":
        error_handling_test(args.url)
    
    # If custom queries are provided, run them
    if args.queries:
        print("\n===== Custom Query Test =====")
        try:
            result = send_request(args.url, args.queries, args.topk, not args.no_scores)
            print("✅ Custom query successful!")
            print_results(result, args.queries, not args.no_scores)
        except Exception as e:
            print(f"❌ Custom query failed: {e}")

if __name__ == "__main__":
    main()