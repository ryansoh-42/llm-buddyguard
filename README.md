# LLM BuddyGuard

An AI tutoring system for Singapore O-Level students using fine-tuned subject-specific models with educational guardrails.

## Project Overview

This project builds an educational AI tutor that:
- Uses fine-tuned HuggingFace models for Physics, Chemistry, and Biology
- Implements local model inference for better performance and privacy
- Provides subject-specific educational guidance with proper O-Level curriculum alignment
- Implements educational guardrails for safe student interaction
- Supports memory-optimized loading for Apple Silicon and other hardware

**Target Subjects:** Physics, Chemistry, Biology (Singapore O-Level curriculum)

## Recent Updates (Latest Commit)

### Implemented Subject-Specific Local Model Loading
- **Fine-tuned Models**: Added support for HuggingFace fine-tuned models:
  - `Fawl/is469_project_physics` (Physics model)
  - `Fawl/is469_project_chem` (Chemistry model)
  - `Fawl/is469_project_bio` (Biology model)
- **Subject-Specific Prompts**: Created optimized system prompts for each subject
- **Memory Optimization**: Configurable model loading to prevent memory overflow
- **Apple Silicon Support**: MPS device optimization with cache clearing
- **Improved Generation**: Better temperature and token settings for focused responses

## Setup and Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Git
- Virtual environment (recommended)
- ~15-20GB free disk space (for all fine-tuned models)
- Internet connection for initial model downloads
- **For Apple Silicon**: 16GB+ unified memory recommended for loading multiple models

### 1. Clone and Setup Environment

```bash
git clone https://github.com/ryansoh-42/llm-buddyguard.git
cd llm-buddyguard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Guardrails AI Validators

The project uses Guardrails AI for content safety. Install the required validators:

```bash
# Create Guardrails AI account and retrieve API key
guardrails configure


# Install Guardrails AI validators
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/profanity_free
guardrails hub install hub://tryolabs/restricttotopic
guardrails hub install hub://guardrails/ban_list
```

These validators provide:
- **toxic_language**: Detects toxic, harmful, or inappropriate language
- **detect_pii**: Identifies personally identifiable information (emails, phone numbers, etc.)
- **profanity_free**: Filters profane or offensive language
- **restricttotopic**: Ensures content stays within allowed educational topics
- **ban_list**: Blocks explicitly banned terms and phrases

### 3. Install Models Locally

The project uses fine-tuned models that need to be downloaded locally:

#### Step 3.1: Hugging Face Authentication
```bash
# Install HF CLI (if not already installed)
pip install huggingface_hub

# Login with your token (get from https://huggingface.co/settings/tokens)
huggingface-cli login
```

#### Step 3.2: Download Fine-tuned Models
The models will be automatically downloaded when first loaded. To pre-download:

```bash
# Download all models (requires ~15-20GB)
python scripts/preload_models.py

# Or download individually in Python:
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
models = ['Fawl/is469_project_physics', 'Fawl/is469_project_chem', 'Fawl/is469_project_bio']
for model in models:
    print(f'Downloading {model}...')
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(model)
    print(f'✅ {model} downloaded')
"
```

### 4. Configure Subject Models

#### Enable/Disable Specific Subjects
In `app.py`, you can configure which models to load:

```python
# To enable Physics model:
try:
    physics_model = FineTunedModel(model_name="Fawl/is469_project_physics")
    subject_models["Physics"] = physics_model
    print("✅ Physics model (local) ready")
except Exception as e:
    print(f"❌ Physics model failed: {e}")
    subject_models["Physics"] = None

# To disable a model (save memory):
subject_models["Physics"] = None
print("ℹ️ Physics model disabled for testing")
```

**Memory Optimization**: For systems with limited memory, load one model at a time:
- **Single model**: ~6-8GB memory usage
- **Two models**: ~12-14GB memory usage  
- **All three models**: ~18-20GB memory usage

### 5. Run the Application

```bash
# Start Streamlit app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 6. Test the System

1. **Select Subject**: Choose Physics, Chemistry, or Biology in the sidebar
2. **Ask Questions**: Try subject-specific questions:
   - Chemistry: "What is H2O?" or "Explain chemical bonding"
   - Physics: "What is Newton's second law?" or "Explain gravity"  
   - Biology: "What is photosynthesis?" or "Explain cell division"
3. **Verify Responses**: Check that answers are subject-appropriate and educational

## Usage Instructions

### Memory Management
- **For 16GB+ systems**: Can load all three models simultaneously
- **For 8-16GB systems**: Load 1-2 models at a time
- **For <8GB systems**: Load one model only, consider using CPU inference

### Model Configuration
Edit `app.py` to uncomment/comment specific subjects:

```python
# Enable Chemistry model (uncomment):
try:
    chemistry_model = FineTunedModel(model_name="Fawl/is469_project_chem")
    subject_models["Chemistry"] = chemistry_model
    print("✅ Chemistry model (local) ready")
except Exception as e:
    print(f"❌ Chemistry model failed: {e}")
    subject_models["Chemistry"] = None

# Disable Chemistry model (comment out and set to None):
# try:
#     chemistry_model = FineTunedModel(model_name="Fawl/is469_project_chem") 
#     subject_models["Chemistry"] = chemistry_model
#     print("✅ Chemistry model (local) ready")
# except Exception as e:
#     print(f"❌ Chemistry model failed: {e}")
#     subject_models["Chemistry"] = None
subject_models["Chemistry"] = None
print("ℹ️ Chemistry model disabled for testing")
```


## Demo Scripts

Understand how the metrics system works by running these executable examples:

### Compare Educational vs API Metrics
See the difference between pedagogical quality scores and technical accuracy metrics:

```bash
python compare_metrics_demo.py
```

**Shows:**
- Educational metrics (step-by-step, tone, no direct answers)
- API metrics (ROUGE, keyword recall, order correctness)
- When to use each metric system

**Example Output:**
```
📚 CURRENT EDUCATIONAL METRICS (ModelEvaluator)
  no_direct_answer: 1.0
  step_by_step_score: 0.67
  tone_score: 0.75
  
🔬 API METRICS (ResponseMetrics - Test API)
  rouge_L_f1: 0.85
  keyword_recall: 0.83
  order_score: 0.95
```

### Chemistry Metrics Examples
Explore how metrics evaluate chemistry-specific responses:

```bash
python demo_chemistry_metrics.py
```

**Demonstrates:**
- Chemical equation balancing evaluation
- Molecule structure explanation scoring
- MCQ answer extraction and accuracy
- Key chemistry terminology detection

**Test Cases Included:**
1. **Balancing Equations** - Tests ROUGE, keyword recall, and reasoning order
2. **Molecule Structure** - Evaluates chemistry terminology and step-by-step approach
3. **MCQ Questions** - Demonstrates answer extraction and exact match scoring

**Example Output:**
```
🧪 CHEMISTRY METRICS DEMO

📝 TEST 1: Balancing Chemical Equations
  ROUGE-L F1: 0.62
  Keyword Recall: 0.83
  Order Score: 0.75
  ⚠️  Order Issue: "First action should be 'count'"

🔬 TEST 2: Molecule Structure
  ROUGE-L F1: 0.89
  Keyword Recall: 1.0
  Order Score: 0.95
  ✅ Order Status: Correct reasoning sequence

❓ TEST 3: Multiple Choice Question
  Exact Match: True
  Extracted Answer: 'B'
  Accuracy: 1.0
```

## Future Improvements
- Add automatic subject detection from question content
- Implement model routing based on question analysis
- Add subject switching recommendations
- Improve cross-subject knowledge integration
- Add confidence scoring for subject-specific responses

## Troubleshooting

### Common Issues

**Error: "Model not found" or download issues**
- Check HuggingFace authentication: `huggingface-cli whoami`
- Ensure internet connection for initial downloads
- Clear cache if needed: `huggingface-cli delete-cache`

**Memory Issues (OOM)**
- Reduce number of loaded models in `app.py`
- Use CPU inference if GPU memory is limited
- Close other memory-intensive applications

**Slow Generation**
- Check if MPS/CUDA is being used: Look for "Model loaded on mps:0" in logs
- Reduce `max_new_tokens` in generation calls
- Lower `temperature` for more focused responses

**Wrong Subject Responses**
- Verify correct model is selected in Streamlit sidebar
- Check that the intended subject model is actually loaded (look for "✅ [Subject] model ready" in terminal)
- Clear browser cache and refresh Streamlit app

### Performance Notes
- **First model load**: Downloads ~5-7GB per model
- **Apple Silicon M4 (18GB)**: Can handle 2-3 models simultaneously
- **Intel/AMD systems**: Performance varies, recommend starting with one model
- **Generation time**: 5-15 seconds per response depending on hardware

## Project Structure

```
llm-buddyguard/
├── app.py                          # Main Streamlit application with dual metrics UI
├── api.py                          # FastAPI server for metrics evaluation
├── automated_eval.py               # Automated evaluation with HF batching & MCQ metrics
├── compare_metrics_demo.py         # Demo comparing educational vs API metrics
├── demo_chemistry_metrics.py       # Chemistry-specific metrics demonstration
├── data/
│   ├── additional_filter.py        # Content filtering for sensitive/irrelevant topics
│   ├── document_process.py         # CSV/document parsing and preprocessing
│   ├── keyword_extraction.py       # Extract key terms from educational content
│   └── sort_documents.py           # Categorise PDF files
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── finetune.py             # Fine-tuned HuggingFace model wrapper
│   │   └── frontier.py             # OpenAI GPT-4o integration with streaming
│   │
│   ├── guardrails.py               # Educational content filtering & safety checks
│   ├── evaluation.py               # Educational metrics & response evaluation
│   ├── metrics.py                  # API metrics (ROUGE, Text F1, Order, Keywords)
│   └── utils.py                    # Utility functions & helpers
│
├── scripts/
│   └── preload_models.py           # Preload or verify fine-tuned subject models locally
│
├── finetune/
│   ├── run.py                      # Fine-tuning script (Llama-3.2-1B-Instruct)
│   ├── data_handling.py            # CSV processing for training data
│   └── data/                       # Training datasets (bio, chemistry, physics)
│
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── API_README.md                   # FastAPI metrics server documentation
```