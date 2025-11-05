#!/usr/bin/env python3
"""
Test dynamic concept extraction for order scoring.
Tests without spaCy to avoid numpy compatibility issues.
"""

import re
import Levenshtein
from typing import List

def extract_concepts_simple(text: str) -> List[str]:
    """
    Simple fallback extraction (no spaCy needed).
    Extracts words longer than 3 characters, filtering out filler words.
    """
    FILLER_WORDS = {
        "first", "second", "third", "then", "next", "finally", "after", "before",
        "during", "while", "when", "where", "what", "which", "that", "this",
        "these", "those", "your", "their", "from", "into", "with", "about",
        "also", "will", "would", "should", "could", "must", "need", "followed"
    }

    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    # Remove duplicates and filler words while preserving order
    seen = set()
    return [w for w in words if w not in FILLER_WORDS and not (w in seen or seen.add(w))]

def compute_dynamic_order_score(generated: str, reference: str):
    """
    Compute order score with dynamic concept extraction.
    Concepts are extracted from reference, then matched in generated.
    """
    # Step 1: Extract concepts from reference (what we're looking for)
    ref_concepts = extract_concepts_simple(reference)

    # Step 2: Extract concepts from generated
    gen_concepts = extract_concepts_simple(generated)

    # Step 3: Find which reference concepts appear in generated
    matched_in_generated = [c for c in ref_concepts if c in gen_concepts]

    if not ref_concepts:
        return {
            'order_score': 0.0,
            'message': 'No concepts in reference',
            'reference_order': [],
            'generated_order': gen_concepts
        }

    if not matched_in_generated:
        return {
            'order_score': 0.0,
            'message': 'No reference concepts found in generated',
            'reference_order': ref_concepts,
            'generated_order': gen_concepts,
            'matched_concepts': [],
            'missing_concepts': ref_concepts
        }

    # Compute edit distance
    edit_dist = Levenshtein.distance(ref_concepts, matched_in_generated)
    normalized_dist = edit_dist / len(ref_concepts)
    order_score = 1.0 - normalized_dist

    correct_order = (ref_concepts == matched_in_generated)
    missing_concepts = [c for c in ref_concepts if c not in matched_in_generated]

    return {
        'order_score': round(order_score, 4),
        'reference_order': ref_concepts,
        'generated_order': gen_concepts,
        'matched_concepts': matched_in_generated,
        'missing_concepts': missing_concepts,
        'edit_distance': edit_dist,
        'correct_order': correct_order,
        'message': 'Order matches' if correct_order else f'{edit_dist} edit(s) needed'
    }

# Test cases
print("=== Dynamic Order Scoring Tests ===\n")

# Test 1: Math - Perfect order
print("Test 1: Math Question - Perfect Order")
ref = "First, factorize the expression. Then multiply the factors. Finally, simplify the result."
gen = "First, factorize the expression. Then multiply the factors. Finally, simplify the result."
result = compute_dynamic_order_score(gen, ref)
print(f"Reference concepts: {result['reference_order']}")
print(f"Matched concepts: {result['matched_concepts']}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 2: Math - Wrong order
print("Test 2: Math Question - Wrong Order")
ref = "First, factorize the expression. Then multiply the factors. Finally, simplify."
gen = "First, multiply the factors. Then factorize the expression. Finally, simplify."
result = compute_dynamic_order_score(gen, ref)
print(f"Reference concepts: {result['reference_order']}")
print(f"Matched concepts: {result['matched_concepts']}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 3: Biology - Mitosis phases (NEW - not in hardcoded list!)
print("Test 3: Biology - Mitosis Phases (Dynamic extraction)")
ref = "During mitosis, prophase occurs first, then metaphase aligns chromosomes, followed by anaphase separation, and finally telophase."
gen = "In mitosis, metaphase happens first, then prophase, then anaphase, and telophase at the end."
result = compute_dynamic_order_score(gen, ref)
print(f"Reference concepts: {result['reference_order']}")
print(f"Matched concepts: {result['matched_concepts']}")
print(f"Missing concepts: {result.get('missing_concepts', [])}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 4: Chemistry - Completely different (missing concepts)
print("Test 4: Missing Key Concepts")
ref = "First, observe the reaction. Then measure the temperature. Finally, record your findings."
gen = "Just write down what you see."
result = compute_dynamic_order_score(gen, ref)
print(f"Reference concepts: {result['reference_order']}")
print(f"Matched concepts: {result.get('matched_concepts', [])}")
print(f"Missing concepts: {result.get('missing_concepts', [])}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

# Test 5: Physics - Partial match
print("Test 5: Partial Match (some concepts present)")
ref = "Calculate the force, then determine the acceleration, and finally solve for mass."
gen = "First solve for mass, then calculate the force."
result = compute_dynamic_order_score(gen, ref)
print(f"Reference concepts: {result['reference_order']}")
print(f"Matched concepts: {result.get('matched_concepts', [])}")
print(f"Missing concepts: {result.get('missing_concepts', [])}")
print(f"Order Score: {result['order_score']} - {result['message']}\n")

print("✅ Dynamic order scoring tests completed!")
print("\nKey improvement: Now extracts ANY concepts from reference (prophase, metaphase, etc.)")
print("No longer limited to hardcoded verb list!")
