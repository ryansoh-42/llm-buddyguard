#!/usr/bin/env python3
"""
Demo: Compare Current vs API Metrics Side by Side
Shows what each evaluation system provides
"""

from src.evaluation import ModelEvaluator
from src.metrics import ResponseMetrics

# Test response
test_response = """To balance the chemical equation H2 + O2 → H2O, first count the atoms on each side. 
On the left: 2 hydrogen atoms and 2 oxygen atoms. On the right: 2 hydrogen atoms and 1 oxygen atom.
Then, add coefficients to balance: 2H2 + O2 → 2H2O. Finally, verify by counting again."""

test_reference = "First count atoms on each side, then add coefficients to balance, finally verify the equation."
test_keywords = ["count", "atoms", "coefficients", "balance", "equation", "verify"]

print("🔬 CHEMISTRY QUESTION METRICS COMPARISON")
print("=" * 80)
print(f"Response: {test_response[:100]}...")
print(f"Reference: {test_reference}")
print(f"Expected Keywords: {test_keywords}")
print("\n" + "=" * 80)

# 1. Current Educational Metrics (ModelEvaluator)
print("\n📚 CURRENT EDUCATIONAL METRICS (ModelEvaluator)")
print("-" * 50)
evaluator = ModelEvaluator()
edu_metrics = evaluator.evaluate_response(
    response=test_response,
    expected_keywords=test_keywords,
    reference_answer=test_reference
)

for key, value in edu_metrics.items():
    if isinstance(value, dict):
        print(f"  {key}: [complex object]")
    else:
        print(f"  {key}: {value}")

# 2. API Metrics (ResponseMetrics - what test API uses)
print("\n\n🔬 API METRICS (ResponseMetrics - Test API)")
print("-" * 50)
api_metrics = ResponseMetrics()
api_result = api_metrics.compute_all_metrics(
    generated=test_response,
    reference=test_reference,
    expected_keywords=test_keywords
)

for key, value in api_result.items():
    if key == "rouge":
        print(f"  {key}:")
        for rouge_type, scores in value.items():
            print(f"    {rouge_type}_f1: {scores['fmeasure']:.3f}")
    elif key == "keyword_recall":
        print(f"  {key}: {value['recall']:.3f} (matched: {value['matched_keywords']})")
    elif key == "order":
        print(f"  {key}: {value['order_score']:.3f} ({value['message']})")
    else:
        print(f"  {key}: {value}")

print("\n\n🎯 KEY DIFFERENCES:")
print("-" * 50)
print("Educational Metrics focus on:")
print("  ✅ Pedagogical quality (step-by-step, tone, no direct answers)")
print("  ✅ Educational appropriateness")
print("  ✅ Overall teaching effectiveness")

print("\nAPI Metrics focus on:")
print("  ✅ Content accuracy (ROUGE similarity)")
print("  ✅ Technical terminology (keyword coverage)")
print("  ✅ Reasoning sequence (order correctness)")
print("  ✅ Response quality (length, word count)")

print("\n💡 RECOMMENDATION:")
print("Show BOTH metrics to get complete picture:")
print("  📚 Educational tab: Teaching quality scores")
print("  🔬 Technical tab: Content accuracy scores")