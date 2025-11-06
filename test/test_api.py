#!/usr/bin/env python3
"""
Test script for Metrics API
Run this after starting the API server with: python api.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_basic_evaluation():
    """Test basic evaluation with ROUGE only"""
    print("=" * 60)
    print("TEST 2: Basic Evaluation (ROUGE only)")
    print("=" * 60)

    payload = {
        "generated": "To solve x² + 5x + 6, factorize by finding two numbers that multiply to 6",
        "reference": "Factorize x² + 5x + 6 to get (x+2)(x+3)"
    }

    response = requests.post(f"{BASE_URL}/evaluate", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        metrics = result["metrics"]

        print(f"Response Length: {metrics['response_length']}")
        print(f"Word Count: {metrics['word_count']}")
        print(f"ROUGE-1 F1: {metrics['rouge']['rouge1']['fmeasure']}")
        print(f"ROUGE-L F1: {metrics['rouge']['rougeL']['fmeasure']}\n")
    else:
        print(f"Error: {response.text}\n")


def test_with_keywords():
    """Test evaluation with keyword recall"""
    print("=" * 60)
    print("TEST 3: With Keyword Recall")
    print("=" * 60)

    payload = {
        "generated": "To solve this, factorize the expression and multiply the terms",
        "expected_keywords": ["factorize", "multiply", "simplify"]
    }

    response = requests.post(f"{BASE_URL}/evaluate", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        metrics = result["metrics"]

        kr = metrics['keyword_recall']
        print(f"Keyword Recall: {kr['recall']}")
        print(f"Matched: {kr['matched_keywords']}")
        print(f"Missing: {kr['missing_keywords']}\n")
    else:
        print(f"Error: {response.text}\n")


def test_order_scoring():
    """Test order scoring"""
    print("=" * 60)
    print("TEST 4: Order Scoring (Biology)")
    print("=" * 60)

    payload = {
        "generated": "During mitosis, metaphase happens first, then prophase, then anaphase",
        "reference": "During mitosis, prophase occurs first, then metaphase, then anaphase"
    }

    response = requests.post(f"{BASE_URL}/evaluate", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        order = result["metrics"]["order"]

        print(f"Order Score: {order['order_score']}")
        print(f"Reference Order: {order['reference_order']}")
        print(f"Generated Order: {order['generated_order']}")
        print(f"Matched: {order['matched_concepts']}")
        print(f"Missing: {order['missing_concepts']}")
        print(f"Message: {order['message']}\n")
    else:
        print(f"Error: {response.text}\n")


def test_mcq():
    """Test MCQ evaluation"""
    print("=" * 60)
    print("TEST 5: MCQ Evaluation")
    print("=" * 60)

    payload = {
        "generated": "The answer is B",
        "reference": "B",
        "is_mcq": True
    }

    response = requests.post(f"{BASE_URL}/evaluate", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        exact_match = result["metrics"]["exact_match"]

        print(f"Exact Match: {exact_match['exact_match']}")
        print(f"Extracted Answer: {exact_match['extracted_answer']}")
        print(f"Accuracy: {exact_match['accuracy']}\n")
    else:
        print(f"Error: {response.text}\n")


def test_full_evaluation():
    """Test full evaluation with all metrics"""
    print("=" * 60)
    print("TEST 6: Full Evaluation (All Metrics)")
    print("=" * 60)

    payload = {
        "generated": "First multiply the terms, then factorize, then simplify",
        "reference": "First factorize, then multiply, then simplify",
        "expected_keywords": ["factorize", "multiply", "simplify"]
    }

    response = requests.post(f"{BASE_URL}/evaluate", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        metrics = result["metrics"]

        print("Metrics Computed:")
        print(f"  - ROUGE-L F1: {metrics['rouge']['rougeL']['fmeasure']}")
        print(f"  - Keyword Recall: {metrics['keyword_recall']['recall']}")
        print(f"  - Order Score: {metrics['order']['order_score']}")
        print(f"  - Order Message: {metrics['order']['message']}")
        print(f"  - API Message: {result['message']}\n")
    else:
        print(f"Error: {response.text}\n")


def test_batch():
    """Test batch evaluation"""
    print("=" * 60)
    print("TEST 7: Batch Evaluation")
    print("=" * 60)

    payload = [
        {
            "generated": "First response",
            "reference": "Correct answer 1"
        },
        {
            "generated": "Second response",
            "reference": "Correct answer 2"
        }
    ]

    response = requests.post(f"{BASE_URL}/evaluate/batch", json=payload)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Total: {result['total']}")
        print(f"Results: {len(result['results'])} items processed\n")
    else:
        print(f"Error: {response.text}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("METRICS API TEST SUITE")
    print("=" * 60)
    print("Make sure the API server is running: python api.py")
    print("=" * 60 + "\n")

    try:
        # Run all tests
        test_health()
        test_basic_evaluation()
        test_with_keywords()
        test_order_scoring()
        test_mcq()
        test_full_evaluation()
        test_batch()

        print("=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("Please start the server first: python api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
