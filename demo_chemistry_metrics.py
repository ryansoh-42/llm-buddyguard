#!/usr/bin/env python3
"""
Demo: Chemistry Metrics for LLM BuddyGuard
Shows how the metrics evaluate chemistry responses
"""

from src.metrics import ResponseMetrics
import json

# Initialize metrics
metrics = ResponseMetrics()

print("🧪 CHEMISTRY METRICS DEMO")
print("=" * 60)

# Test Case 1: Balancing Chemical Equations
print("\n📝 TEST 1: Balancing Chemical Equations")
print("-" * 40)

reference = "First, count the atoms on each side. Then, add coefficients to balance. Finally, verify the equation."
generated = "First, add coefficients randomly. Then, count atoms. Finally, hope it's balanced."

result = metrics.compute_all_metrics(
    generated=generated,
    reference=reference,
    expected_keywords=["count", "atoms", "coefficients", "balance", "equation"]
)

print(f"Reference: {reference}")
print(f"Generated: {generated}")
print(f"\n📊 SCORES:")
print(f"  ROUGE-L F1: {result['rouge']['rougeL']['fmeasure']:.3f}")
print(f"  Keyword Recall: {result['keyword_recall']['recall']:.3f}")
print(f"  Order Score: {result['order']['order_score']:.3f}")
print(f"  ⚠️  Order Issue: {result['order']['message']}")

# Test Case 2: Molecule Structure
print("\n\n🔬 TEST 2: Molecule Structure")
print("-" * 40)

reference = "To understand H2O structure: first identify the central atom (oxygen), then count valence electrons, finally draw bonds."
generated = "H2O has oxygen in the middle, with hydrogen atoms bonded to it. Count electrons and draw the structure."

result2 = metrics.compute_all_metrics(
    generated=generated,
    reference=reference,
    expected_keywords=["oxygen", "hydrogen", "electrons", "bonds", "structure"]
)

print(f"Reference: {reference}")
print(f"Generated: {generated}")
print(f"\n📊 SCORES:")
print(f"  ROUGE-L F1: {result2['rouge']['rougeL']['fmeasure']:.3f}")
print(f"  Keyword Recall: {result2['keyword_recall']['recall']:.3f}")
print(f"  Order Score: {result2['order']['order_score']:.3f}")
print(f"  ✅ Order Status: {result2['order']['message']}")

# Test Case 3: MCQ Question
print("\n\n❓ TEST 3: Multiple Choice Question")
print("-" * 40)

reference = "B"
generated = "The answer is B because oxygen has 8 electrons."

result3 = metrics.compute_all_metrics(
    generated=generated,
    reference=reference,
    is_mcq=True
)

print(f"Correct Answer: {reference}")
print(f"Generated: {generated}")
print(f"\n📊 SCORES:")
print(f"  Exact Match: {result3['exact_match']['exact_match']}")
print(f"  Extracted Answer: '{result3['exact_match']['extracted_answer']}'")
print(f"  Accuracy: {result3['exact_match']['accuracy']:.3f}")

# Summary
print("\n\n🎯 SUMMARY FOR CHEMISTRY")
print("=" * 60)
print("The metrics system evaluates:")
print("✅ Content accuracy (ROUGE scores)")
print("✅ Key chemistry terms usage")
print("✅ Correct reasoning order (crucial for chemistry steps)")
print("✅ MCQ answer extraction")
print("✅ Response quality metrics")
print("\n💡 This helps ensure your chemistry model gives:")
print("   - Scientifically accurate responses")
print("   - Proper step-by-step reasoning")
print("   - Appropriate chemistry terminology")