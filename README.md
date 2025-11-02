# LLM BuddyGuard

An AI tutoring system for Singapore O-Level students using open-weight baseline models and frontier model comparison.

## Project Overview

This project builds an educational AI tutor that:
- Tests baseline performance of open-weight LLMs (no fine-tuning)
- Compares against frontier models (GPT-4o)
- Implements educational guardrails
- Provides step-by-step tutoring without direct answers

**Target Subjects:** Mathematics, Science, English (Singapore O-Level)

## Setup and Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Git
- Virtual environment (recommended)

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

### 2. Hugging Face Authentication (Required for Llama Models)

The project uses Meta's Llama-3.2-3B-Instruct model, which requires authentication:

#### Step 2.1: Get Hugging Face Token
1. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "llm-buddyguard-project")
4. Select "Read" permissions
5. Generate and copy the token

#### Step 2.2: Request Llama Model Access
1. Visit [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. Click "Request access"
3. Fill out Meta's form with your intended use case
4. Accept the license terms
5. Submit and wait for approval (usually within minutes to hours)

#### Step 2.3: Login via CLI
```bash
# Install HF CLI (if not already installed)
pip install huggingface_hub

# Login with your token
huggingface-cli login
# Enter your token when prompted
```

#### Step 2.4: Verify Access
```bash
# Check authentication status
huggingface-cli whoami

# Test model access
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Testing access...'); info = api.model_info('meta-llama/Llama-3.2-3B-Instruct'); print('✅ Access granted!')"
```

### 3. OpenAI API Setup (Optional - for Frontier Model Comparison)

If you want to compare with GPT-4o:

1. Get an OpenAI API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root:
```bash
cp .envexample .env
```
3. Add your API key to `.env`:
```
OPENAI_API_KEY="your-api-key-here"
```

### 4. Test Installation

```bash
# Test baseline model loading
python -c "from src.models.baseline import BaselineModel; model = BaselineModel(); print('✅ Baseline model loaded successfully!')"

# Run the Streamlit app
streamlit run app.py
```

## Troubleshooting

### Common Issues

**Error: "Cannot access gated repo"**
- Ensure you've requested access to the Llama model on Hugging Face
- Check that your access request has been approved
- Verify you're logged in: `huggingface-cli whoami`

**Error: "No module named 'src'"**
- Make sure you're in the project root directory
- Ensure your virtual environment is activated

**Error: CUDA/GPU issues**
- The project works on CPU, though it's slower
- For GPU support, ensure you have CUDA-compatible PyTorch installed

**Token/Authentication issues**
- Try logging out and back in: `huggingface-cli logout && huggingface-cli login`
- Clear HF cache if needed: `huggingface-cli delete-cache`

### Performance Notes
- First model load will download ~6GB of model weights
- CPU inference takes 10-30 seconds per response
- GPU inference (if available) is much faster

## Project Structure
