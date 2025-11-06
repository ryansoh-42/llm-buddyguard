#!/usr/bin/env python3
"""
Standalone test for order scoring functionality.
Tests without importing ROUGE to avoid numpy compatibility issues.
"""

import re
import Levenshtein
from typing import List

# Common reasoning verbs for O-Level subjects
REASONING_VERBS = {
    # Math
    "solve", "calculate", "factorize", "factor", "multiply", "divide",
    "add", "subtract", "simplify", "expand", "substitute", "evaluate",
    "compute", "find", "determine",
    # Science
    "observe", "hypothesize", "test", "experiment", "conclude",
    "analyze", "measure", "record", "identify", "classify",
    # General
    "check", "verify", "prove", "compare", "explain", "describe"
}

def extract_reasoning_steps(text: str) -> List[str]:
    """Extract reasoning steps using keyword matching."""
    steps = []
    for verb in REASONING_VERBS:
        pattern = r'\b' + re.escape(verb) + r'\b'
        match = re.search(pattern, text.lower())
        if match:
            steps.append((match.start(), verb))

    # Sort by position
    steps.sort(key=lambda x: x[0])
    return [verb for _, verb in steps]

def compute_order_score(generated: str, reference: str):
    """Compute order score using Levenshtein distance."""
    ref_steps = extract_reasoning_steps(reference)
    gen_steps = extract_reasoning_steps(generated)

    if not ref_steps or not gen_steps:
        return {
            'order_score': 0.0,
            'reference_order': ref_steps,
            'generated_order': gen_steps,
            'message': 'Missing steps in one or both texts'
        }

    # Compute edit distance
    edit_dist = Levenshtein.distance(ref_steps, gen_steps)
    max_len = max(len(ref_steps), len(gen_steps))
    normalized_dist = edit_dist / max_len
    order_score = 1.0 - normalized_dist

    correct_order = (ref_steps == gen_steps)

    return {
        'order_score': round(order_score, 4),
        'reference_order': ref_steps,
        'generated_order': gen_steps,
        'edit_distance': edit_dist,
        'correct_order': correct_order,
        'message': 'Order matches' if correct_order else f'{edit_dist} edit(s) needed'
    }

# Test cases
print("=== Order Scoring Tests ===\n")

# Test 1: Perfect match
print("Test 1: Perfect Order Match")
ref = "First, factorize the expression. Then multiply the factors. Finally, simplify the result."
gen = "First, factorize the expression. Then multiply the factors. Finally, simplify the result."
result = compute_order_score(gen, ref)
print(f"Reference: {result['reference_order']}")
print(f"Generated: {result['generated_order']}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 2: Swapped order
print("Test 2: Wrong Order (swap multiply and factorize)")
ref = "First, factorize the expression. Then multiply the factors. Finally, simplify."
gen = "First, multiply the factors. Then factorize the expression. Finally, simplify."
result = compute_order_score(gen, ref)
print(f"Reference: {result['reference_order']}")
print(f"Generated: {result['generated_order']}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 3: Completely different
print("Test 3: Missing steps")
ref = "First, observe the phenomenon. Then hypothesize. Finally, test your hypothesis."
gen = "Just observe and record your findings."
result = compute_order_score(gen, ref)
print(f"Reference: {result['reference_order']}")
print(f"Generated: {result['generated_order']}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 4: Extra step
print("Test 4: Extra step in generated")
ref = "Calculate the value and simplify."
gen = "First observe, then calculate the value, multiply by two, and simplify."
result = compute_order_score(gen, ref)
print(f"Reference: {result['reference_order']}")
print(f"Generated: {result['generated_order']}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

print("✅ Order scoring tests completed!")
