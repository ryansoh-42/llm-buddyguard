# Metrics API Documentation

A FastAPI service that evaluates model responses using multiple metrics in one JSON response.

## Features

- ✅ **ROUGE Scores** - Text similarity metrics
- ✅ **Text F1** - Word overlap evaluation (automatic, no keyword list needed)
- ✅ **Keyword Recall** - Concept coverage using simple recall metric
- ✅ **Exact Match** - MCQ answer grading
- ✅ **Order Scoring** - Dynamic reasoning step validation (always applied)
- ✅ **Batch Evaluation** - Process multiple responses at once

---

## Installation

### Option 1: Install All Dependencies (Recommended)

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Download spaCy language model (required for order scoring)
python -m spacy download en_core_web_sm
```

### Option 2: Install Only Metrics API Dependencies

```bash
# Install minimal dependencies for metrics API only
pip install fastapi uvicorn pydantic rouge-score python-Levenshtein spacy

# Download spaCy language model (required for order scoring)
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
# Test that metrics work
python -c "from src.metrics import ResponseMetrics; print('✅ Metrics ready!')"
```

---

## Quick Start

### Start the API Server

```bash
# Method 1: Direct run
python api.py

# Method 2: Using uvicorn
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: **http://localhost:8000**

---

## API Endpoints

### 1. **GET /** - Root Information
```bash
curl http://localhost:8000/
```

### 2. **GET /health** - Health Check
```bash
curl http://localhost:8000/health
```

### 3. **GET /metrics-info** - Available Metrics Info
```bash
curl http://localhost:8000/metrics-info
```

### 4. **POST /evaluate** - Main Evaluation Endpoint

**Request:**
```json
{
  "generated": "string (required)",
  "reference": "string (optional)",
  "expected_keywords": ["keyword1", "keyword2"] (optional),
  "is_mcq": false (optional)
}
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "response_length": 123,
    "word_count": 45,
    "rouge": {...},
    "text_f1": {...},
    "order": {...},
    "keyword_recall": {...},
    "exact_match": {...}
  },
  "message": "All applicable metrics computed successfully"
}
```

**Notes:**
- When reference is provided, `text_f1` and `order` are **always computed automatically**
- The `message` field provides feedback about which metrics were computed or skipped

---

## Usage Examples

### Example 1: Basic Evaluation (ROUGE + Text F1)

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "To solve x² + 5x + 6, factorize by finding two numbers that multiply to 6",
    "reference": "Factorize x² + 5x + 6 to get (x+2)(x+3)"
  }'
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "response_length": 73,
    "word_count": 15,
    "rouge": {
      "rouge1": {"precision": 0.5333, "recall": 0.6154, "fmeasure": 0.5714},
      "rouge2": {"precision": 0.2857, "recall": 0.3077, "fmeasure": 0.2963},
      "rougeL": {"precision": 0.5333, "recall": 0.6154, "fmeasure": 0.5714}
    },
    "text_f1": {
      "precision": 0.5333,
      "recall": 0.6154,
      "f1": 0.5714,
      "matched_words": ["factorize", "x", "5", "6", "to", "get", "2", "3"],
      "missing_words": ["x+2", "x+3"],
      "extra_words": ["solve", "by", "finding", "two", "numbers", "that", "multiply"]
    },
    "order": {
      "order_score": 0.6667,
      "correct_order": true,
      "message": "Order matches"
    }
  },
  "message": "All applicable metrics computed successfully"
}
```

---

### Example 2: Keyword Coverage (Chemistry)

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "Chlorine is more reactive than bromine and can displace it from solution",
    "expected_keywords": ["chlorine", "bromine", "reactive", "displace", "halogen"]
  }'
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "response_length": 72,
    "word_count": 12,
    "keyword_recall": {
      "recall": 0.80,
      "matched_keywords": ["chlorine", "bromine", "reactive", "displace"],
      "missing_keywords": ["halogen"],
      "match_count": 4,
      "expected_count": 5
    }
  },
  "message": "No reference provided - skipping ROUGE, Text F1, and Order metrics"
}
```

---

### Example 3: Order Scoring (Biology)

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "During mitosis, metaphase happens first, then prophase, then anaphase",
    "reference": "During mitosis, prophase occurs first, then metaphase, then anaphase"
  }'
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "response_length": 70,
    "word_count": 11,
    "rouge": {...},
    "order": {
      "order_score": 0.5556,
      "reference_order": ["mitosis", "prophase", "occurs", "metaphase", "anaphase"],
      "generated_order": ["mitosis", "metaphase", "happens", "prophase", "anaphase"],
      "matched_concepts": ["mitosis", "metaphase", "prophase", "anaphase"],
      "missing_concepts": ["occurs"],
      "edit_distance": 4,
      "correct_order": false,
      "normalized_distance": 0.4444,
      "message": "4 edit(s) needed for correct order"
    }
  },
  "message": "All applicable metrics computed successfully"
}
```

---

### Example 4: MCQ Evaluation

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "The answer is B",
    "reference": "B",
    "is_mcq": true
  }'
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "response_length": 14,
    "word_count": 4,
    "rouge": {...},
    "exact_match": {
      "exact_match": true,
      "extracted_answer": "b",
      "reference_answer": "b",
      "accuracy": 1.0
    },
    "text_f1": {...},
    "order": {...}
  },
  "message": "All applicable metrics computed successfully"
}
```

---

### Example 5: Full Evaluation (All Metrics)

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "First multiply the terms, then factorize, then simplify",
    "reference": "First factorize, then multiply, then simplify",
    "expected_keywords": ["multiply", "factorize", "simplify"]
  }'
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "response_length": 56,
    "word_count": 9,
    "rouge": {
      "rouge1": {"precision": 0.7778, "recall": 0.7778, "fmeasure": 0.7778},
      "rouge2": {"precision": 0.5, "recall": 0.5, "fmeasure": 0.5},
      "rougeL": {"precision": 0.7778, "recall": 0.7778, "fmeasure": 0.7778}
    },
    "text_f1": {
      "precision": 0.7778,
      "recall": 0.7778,
      "f1": 0.7778,
      "matched_words": ["first", "multiply", "then", "factorize", "simplify"],
      "missing_words": [],
      "extra_words": ["the", "terms"]
    },
    "order": {
      "order_score": 0.3333,
      "reference_order": ["factorize", "multiply", "simplify"],
      "generated_order": ["multiply", "terms", "factorize", "simplify"],
      "matched_concepts": ["multiply", "factorize", "simplify"],
      "missing_concepts": [],
      "edit_distance": 2,
      "correct_order": false,
      "message": "2 edit(s) needed for correct order"
    },
    "keyword_recall": {
      "recall": 1.0,
      "matched_keywords": ["multiply", "factorize", "simplify"],
      "missing_keywords": [],
      "match_count": 3,
      "expected_count": 3
    }
  },
  "message": "All applicable metrics computed successfully"
}
```

---

### Example 6: Batch Evaluation

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "generated": "First response",
      "reference": "Correct answer 1"
    },
    {
      "generated": "Second response",
      "reference": "Correct answer 2"
    }
  ]'
```

---

## Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/evaluate"

# Request payload (no need to escape long text!)
payload = {
    "generated": "First multiply, then factorize, then simplify",
    "reference": "First factorize, then multiply, then simplify",
    "expected_keywords": ["multiply", "factorize", "simplify"]
}

# Make request
response = requests.post(url, json=payload)

# Get metrics
if response.status_code == 200:
    result = response.json()
    metrics = result["metrics"]

    print(f"Order Score: {metrics['order']['order_score']}")
    print(f"Text F1: {metrics['text_f1']['f1']}")
    print(f"Keyword Recall: {metrics['keyword_recall']['recall']}")
    print(f"ROUGE-L: {metrics['rouge']['rougeL']['fmeasure']}")
else:
    print(f"Error: {response.status_code}")
```

---

## Quick Endpoints

### ROUGE Only
```bash
curl -X POST http://localhost:8000/evaluate/rouge \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "student answer",
    "reference": "correct answer"
  }'
```

### Order Scoring Only
```bash
curl -X POST http://localhost:8000/evaluate/order \
  -H "Content-Type: application/json" \
  -d '{
    "generated": "student answer",
    "reference": "correct answer",
    "use_spacy": true
  }'
```

---

## Interactive API Documentation

FastAPI provides automatic interactive docs:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Response Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 422 | Validation Error (invalid request body) |
| 500 | Internal Server Error |

---

## Metric Requirements

| Metric | Requires | Optional |
|--------|----------|----------|
| **ROUGE** | `generated`, `reference` | - |
| **Text F1** | `generated`, `reference` | - |
| **Keyword Recall** | `generated`, `expected_keywords` | - |
| **Exact Match** | `generated`, `reference`, `is_mcq=true` | - |
| **Order Score** | `generated`, `reference` | `use_spacy` |
| **Basic Stats** | `generated` | - |

---

## Notes

- The API automatically computes only applicable metrics based on provided inputs
- If `reference` is provided, ROUGE, text F1, and order scoring are **always computed automatically**
- If `expected_keywords` is provided, keyword recall will be computed
- Basic stats (length, word count) are always computed
- Set `is_mcq=true` for MCQ evaluation (requires `reference`)
- For long text, use Python client to avoid JSON escaping issues

---

## Production Deployment

```bash
# Production server with Gunicorn
pip install gunicorn

gunicorn api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Error Handling

All errors return:
```json
{
  "detail": "Error message here"
}
```

---

## License

Same as main project
