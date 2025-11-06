# Metrics Service Documentation

## Overview

`metrics.py` provides a standalone service for evaluating model response quality. It computes ROUGE scores, Text F1 (word overlap), Keyword Recall (concept coverage), exact match accuracy, and order scoring.

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

#### 2. `compute_text_f1(generated, reference, case_sensitive=False)`

Computes F1 score based on word overlap between generated and reference text. Automatically compares all words - no manual keyword list needed.

**Input:**
- `generated` (str): Model-generated response
- `reference` (str): Reference/correct answer
- `case_sensitive` (bool, optional): Match case-sensitively (default: False)

**Output:**
```python
{
    'precision': 0.3333,  # 3 matched / 9 generated words
    'recall': 0.5000,     # 3 matched / 6 reference words
    'f1': 0.4000,
    'matched_words': ['mitosis', 'prophase', 'metaphase'],
    'missing_words': ['anaphase', 'stages', 'involves'],
    'extra_words': ['cell', 'divides', 'during', 'through', 'and'],
    'generated_word_count': 9,
    'reference_word_count': 6
}
```

**How it works:**
- Tokenizes both texts into words
- Uses multisets (Counter) to handle repeated words correctly
- Calculates precision, recall, F1 based on word overlap
- Returns detailed analysis of matched, missing, and extra words

---

#### 3. `compute_keyword_recall(generated, expected_keywords, case_sensitive=False)`

Computes keyword coverage using simple recall metric. Measures what percentage of expected keywords appear in the response.

**Input:**
- `generated` (str): Model-generated response
- `expected_keywords` (List[str]): Keywords that should appear
- `case_sensitive` (bool, optional): Match case-sensitively (default: False)

**Output:**
```python
{
    'recall': 0.80,          # 4 out of 5 keywords found
    'matched_keywords': ['chlorine', 'bromine', 'reactive', 'displace'],
    'missing_keywords': ['halogen'],
    'match_count': 4,
    'expected_count': 5
}
```

**Why recall instead of F1?**
- **Recall** directly answers: "What % of required concepts did the student mention?"
- F1 gets artificially low when response is long (low precision), even if all keywords present
- For concept coverage, recall is the most meaningful metric

---

#### 4. `compute_exact_match(generated, reference, normalize=True)`

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

#### 5. `compute_order_score(generated, reference, use_spacy=True)`

Evaluates whether concepts appear in correct order using **dynamic concept extraction**.

**Input:**
- `generated` (str): Model-generated response
- `reference` (str): Reference response with correct concept order
- `use_spacy` (bool, optional): Use spaCy NLP for extraction (default: True)

**Output:**
```python
{
    'order_score': 0.5556,  # 0-1, higher = better order match
    'reference_order': ['mitosis', 'prophase', 'occurs', 'metaphase', 'aligns', 'chromosomes', 'anaphase', 'separation', 'telophase'],
    'generated_order': ['mitosis', 'prophase', 'metaphase', 'anaphase', 'telophase'],
    'matched_concepts': ['mitosis', 'prophase', 'metaphase', 'anaphase', 'telophase'],
    'missing_concepts': ['occurs', 'aligns', 'chromosomes', 'separation'],
    'edit_distance': 4,
    'correct_order': False,
    'message': '4 edit(s) needed for correct order'
}
```

**How it works:**
1. **Dynamically extracts** key concepts (nouns + verbs) from reference text - NO hardcoded lists!
2. Finds those concepts in generated text
3. Compares order using Levenshtein edit distance
4. Returns score where 1.0 = perfect match, 0.0 = completely different

**Key Feature:** Adapts to ANY subject/question:
- Math: "factorize", "multiply", "simplify"
- Biology: "prophase", "metaphase", "anaphase"
- Chemistry: "react", "precipitate", "dissolve"

---

#### 6. `compute_all_metrics(generated, reference=None, expected_keywords=None, is_mcq=False)`

Computes all available metrics in one call.

**Input:**
- `generated` (str): Model-generated response
- `reference` (str, optional): Reference answer for ROUGE/text F1/exact match/order scoring
- `expected_keywords` (List[str], optional): Keywords for coverage evaluation
- `is_mcq` (bool, optional): Whether this is an MCQ (default: False)

**What gets computed:**
- **Always**: `response_length`, `word_count`
- **If reference provided**: `rouge`, `text_f1`, `order` (always automatic)
- **If reference + is_mcq=True**: `exact_match`
- **If expected_keywords provided**: `keyword_recall`

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
    'text_f1': {
        'precision': 0.5000,
        'recall': 0.6667,
        'f1': 0.5714,
        'matched_words': ['factorize', 'multiply', 'x'],
        'missing_words': ['simplify'],
        'extra_words': ['the', 'expression']
    },
    'order': {
        'order_score': 1.0,
        'reference_order': ['factorize', 'multiply', 'simplify'],
        'generated_order': ['factorize', 'multiply', 'simplify'],
        'correct_order': True
    },
    'keyword_recall': {
        'recall': 0.80,
        'matched_keywords': ['chlorine', 'bromine', 'reactive', 'displace'],
        'missing_keywords': ['halogen']
    }
}
```

**Note:** When reference is provided, `text_f1` and `order` scoring are **always computed automatically** - no opt-in required.

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

# With reference answer (includes ROUGE, text F1, and order scoring)
result = metrics.compute_all_metrics(
    generated="To factorize x² + 5x + 6...",
    reference="(x+2)(x+3)"
)
# Includes: rouge, text_f1, order

# With keywords (measures concept coverage)
result = metrics.compute_all_metrics(
    generated="Chlorine is more reactive and displaces bromine",
    expected_keywords=["chlorine", "reactive", "displace", "bromine", "halogen"]
)
# Includes: keyword_recall

# MCQ evaluation
result = metrics.compute_all_metrics(
    generated="The answer is B",
    reference="B",
    is_mcq=True
)

# Full evaluation (all metrics)
result = metrics.compute_all_metrics(
    generated="First multiply, then factorize, then simplify",
    reference="First factorize, then multiply, then simplify",
    expected_keywords=["multiply", "factorize", "simplify"],
    is_mcq=False
)
print(result['order']['order_score'])  # Order score (always computed when reference provided)
print(result['keyword_recall']['recall'])  # Keyword coverage
print(result['text_f1']['f1'])  # Word overlap (always computed when reference provided)
print(result['rouge']['rougeL']['fmeasure'])  # ROUGE-L F1
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

| Metric | Required Input | Optional Input | Notes |
|--------|---------------|----------------|-------|
| ROUGE | `generated`, `reference` | - | Always computed when reference provided |
| Text F1 | `generated`, `reference` | `case_sensitive` | Always computed when reference provided |
| Keyword Recall | `generated`, `expected_keywords` | `case_sensitive` | Only when keywords list provided |
| Exact Match | `generated`, `reference`, `is_mcq=True` | `normalize` | Only for MCQ questions |
| Order Score | `generated`, `reference` | `use_spacy` | Always computed when reference provided |
| All Metrics | `generated` | `reference`, `expected_keywords`, `is_mcq` | Computes applicable metrics based on inputs |

---

## Output Format Summary

All methods return dictionaries with:
- **Numeric scores** (float, 0-1 range)
- **Metadata** (matched keywords, violations, etc.)
- **Flags** (is_safe, exact_match, etc.)

All numeric scores are rounded to 4 decimal places for consistency.

---

## Notes

- **No reference needed**: Basic metrics (length, word count) work without reference answers
- **Partial inputs**: Provide only what you have; missing inputs skip relevant metrics
- **Reusable**: Create one `ResponseMetrics` instance and reuse for multiple evaluations
- **Model-agnostic**: Works with any text generation model (black-box evaluation)
- **Automatic metrics when reference provided**: `text_f1` and `order` scoring are always computed - no configuration needed
- **Keyword recall for concept coverage**: Simple recall metric (% of required keywords present) - more meaningful than F1 for educational use
- **MCQ evaluation**: Set `is_mcq=True` to add exact match metric; other metrics still computed for explanation quality
- **Dynamic concept extraction**: Order scoring adapts to any subject/domain automatically
