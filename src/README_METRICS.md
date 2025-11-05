# Metrics Service Documentation

## Overview

`metrics.py` provides a standalone service for evaluating model response quality. It computes ROUGE scores, Keyword F1, exact match accuracy, and safety scores.

## Quick Start

```python
from src.metrics import ResponseMetrics

metrics = ResponseMetrics()
result = metrics.compute_rouge(
    generated="To factorize x² + 5x + 6, find two numbers that multiply to 6.",
    reference="Factorize x² + 5x + 6 to get (x+2)(x+3)"
)
```

## API Reference

### `ResponseMetrics` Class

#### 1. `compute_rouge(generated, reference)`

Computes ROUGE scores between generated and reference text.

**Input:**
- `generated` (str): Model-generated response
- `reference` (str): Ground truth/reference answer

**Output:**
```python
{
    'rouge1': {'precision': 0.75, 'recall': 0.80, 'fmeasure': 0.77},
    'rouge2': {'precision': 0.50, 'recall': 0.55, 'fmeasure': 0.52},
    'rougeL': {'precision': 0.70, 'recall': 0.75, 'fmeasure': 0.72}
}
```

---

#### 2. `compute_keyword_f1(generated, expected_keywords, case_sensitive=False)`

Computes F1 score based on keyword presence.

**Input:**
- `generated` (str): Model-generated response
- `expected_keywords` (List[str]): Keywords that should appear
- `case_sensitive` (bool, optional): Match case-sensitively (default: False)

**Output:**
```python
{
    'precision': 0.80,
    'recall': 0.75,
    'f1': 0.77,
    'matched_keywords': ['factorize', 'multiply'],
    'missing_keywords': ['divide'],
    'match_count': 2,
    'expected_count': 3
}
```

---

#### 3. `compute_exact_match(generated, reference, normalize=True)`

Computes exact match accuracy (for MCQ answers).

**Input:**
- `generated` (str): Model's answer (e.g., "The answer is B")
- `reference` (str): Correct answer (e.g., "B")
- `normalize` (bool, optional): Normalize text before comparison (default: True)

**Output:**
```python
{
    'exact_match': True,
    'extracted_answer': 'b',
    'reference_answer': 'b',
    'accuracy': 1.0
}
```

---

#### 4. `compute_safety_score(response, violation_patterns=None)`

Checks for safety/guardrail violations.

**Input:**
- `response` (str): Model's response
- `violation_patterns` (List[str], optional): Custom regex patterns for violations

**Output:**
```python
{
    'is_safe': True,
    'safety_score': 1.0,
    'violations': [],
    'violation_count': 0
}
```

---

#### 5. `compute_all_metrics(generated, reference=None, expected_keywords=None, is_mcq=False)`

Computes all available metrics in one call.

**Input:**
- `generated` (str): Model-generated response
- `reference` (str, optional): Reference answer for ROUGE/exact match
- `expected_keywords` (List[str], optional): Keywords for F1
- `is_mcq` (bool, optional): Whether this is an MCQ (default: False)

**Output:**
```python
{
    'response_length': 86,
    'word_count': 21,
    'rouge': {
        'rouge1': {'precision': 0.68, 'recall': 0.68, 'fmeasure': 0.68},
        'rouge2': {'precision': 0.50, 'recall': 0.50, 'fmeasure': 0.50},
        'rougeL': {'precision': 0.68, 'recall': 0.68, 'fmeasure': 0.68}
    },
    'keyword_f1': {
        'precision': 0.20,
        'recall': 1.0,
        'f1': 0.33,
        'matched_keywords': ['factorize', 'multiply', 'add'],
        'missing_keywords': []
    },
    'safety': {
        'is_safe': True,
        'safety_score': 1.0,
        'violations': [],
        'violation_count': 0
    }
}
```

---

## Integration Example

### Basic Usage

```python
from src.metrics import ResponseMetrics

metrics = ResponseMetrics()

# Minimal - only generated response
result = metrics.compute_all_metrics(
    generated="Let's solve this step by step..."
)

# With reference answer
result = metrics.compute_all_metrics(
    generated="To factorize x² + 5x + 6...",
    reference="(x+2)(x+3)"
)

# With keywords
result = metrics.compute_all_metrics(
    generated="To factorize x² + 5x + 6...",
    expected_keywords=["factorize", "multiply", "two numbers"]
)

# MCQ evaluation
result = metrics.compute_all_metrics(
    generated="The answer is B",
    reference="B",
    is_mcq=True
)
```

### Integration with ModelEvaluator

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

metrics = evaluator.evaluate_response(
    response="Model's response here...",
    reference_answer="Correct answer here...",
    expected_keywords=["key", "terms"],
    is_mcq=False
)

# Access ROUGE scores
print(metrics['rouge1_f1'])  # 0.68
print(metrics['keyword_f1_score'])  # 0.33
print(metrics['safety_score'])  # 1.0
```

### Streamlit Integration

```python
import streamlit as st
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Generate response
response = model.generate(user_question)

# Evaluate
metrics = evaluator.evaluate_response(
    response=response["response"],
    reference_answer="Correct answer",
    expected_keywords=["keyword1", "keyword2"]
)

# Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ROUGE-L F1", f"{metrics.get('rougeL_f1', 0):.2f}")
with col2:
    st.metric("Keyword F1", f"{metrics.get('keyword_f1_score', 0):.2f}")
with col3:
    st.metric("Safety", "✓" if metrics.get('safety_score') == 1.0 else "✗")
```

---

## Input Requirements Summary

| Metric | Required Input | Optional Input |
|--------|---------------|----------------|
| ROUGE | `generated`, `reference` | - |
| Keyword F1 | `generated`, `expected_keywords` | `case_sensitive` |
| Exact Match | `generated`, `reference` | `normalize` |
| Safety | `generated` | `violation_patterns` |
| All Metrics | `generated` | `reference`, `expected_keywords`, `is_mcq` |

---

## Output Format Summary

All methods return dictionaries with:
- **Numeric scores** (float, 0-1 range)
- **Metadata** (matched keywords, violations, etc.)
- **Flags** (is_safe, exact_match, etc.)

All numeric scores are rounded to 4 decimal places for consistency.

---

## Notes

- **No reference needed**: Basic metrics (safety, length) work without reference answers
- **Partial inputs**: Provide only what you have; missing inputs skip relevant metrics
- **Reusable**: Create one `ResponseMetrics` instance and reuse for multiple evaluations
- **Model-agnostic**: Works with any text generation model
