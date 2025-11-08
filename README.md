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

### ✅ Implemented Subject-Specific Local Model Loading
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

### 2. Install Models Locally

The project now uses fine-tuned models that need to be downloaded locally:

#### Step 2.1: Hugging Face Authentication
```bash
# Install HF CLI (if not already installed)
pip install huggingface_hub

# Login with your token (get from https://huggingface.co/settings/tokens)
huggingface-cli login
```

#### Step 2.2: Download Fine-tuned Models
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

### 3. Configure Subject Models

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

### 4. Run the Application

```bash
# Start Streamlit app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Test the System

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

## TODO

### Immediate Testing Needed
- [ ] **Cross-subject Question Testing**: Test what happens when you select "Chemistry" but ask Physics questions (e.g., "What is Newton's second law?" while Chemistry is selected)
- [ ] **Subject Specificity**: Verify that models provide appropriate responses when asked off-topic questions
- [ ] **Model Consistency**: Test if Chemistry model gives chemistry-focused answers to general science questions

### Future Improvements
- [ ] Add automatic subject detection from question content
- [ ] Implement model routing based on question analysis
- [ ] Add subject switching recommendations
- [ ] Improve cross-subject knowledge integration
- [ ] Add confidence scoring for subject-specific responses

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
├── app.py                 # Main Streamlit application
├── src/
│   ├── models/
│   │   ├── baseline.py    # Local model loading with subject-specific prompts
│   │   ├── frontier.py    # OpenAI GPT-4o integration
│   │   └── peft.py       # PEFT model support
│   ├── guardrails.py     # Educational content filtering
│   ├── evaluation.py     # Response evaluation metrics
│   └── utils.py         # Utility functions
├── scripts/
│   └── preload_models.py # Model download script
├── finetune/             # Fine-tuning data and scripts
└── test/                # Test files
```
