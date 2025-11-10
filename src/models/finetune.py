# src/models/baseline.py
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
try:
    from transformers import AutoModelForInference
    HAS_INFERENCE_MODEL = True
    print("✅ AutoModelForInference available")
except ImportError:
    HAS_INFERENCE_MODEL = False
    print("ℹ️ AutoModelForInference not available, using AutoModelForCausalLM")
import torch

class FineTunedModel:
    """
    Fine-tuned subject-specific model for O-Level tutoring.
    Loads HuggingFace fine-tuned models with subject-specific prompts.
    """
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto"
    ):
        """
        Initialize baseline model from HuggingFace.
        
        Args:
            model_name: HF model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
            device: Device placement ("auto", "cuda", "cpu")
        """
        print(f"Loading baseline model: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try AutoModelForInference first, fallback to AutoModelForCausalLM
        model_loaded = False
        if HAS_INFERENCE_MODEL:
            try:
                print("🔄 Trying AutoModelForInference (recommended approach)...")
                self.model = AutoModelForInference.from_pretrained(
                    model_name,
                    device_map=device,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                print("✅ AutoModelForInference loaded successfully!")
                model_loaded = True
            except Exception as e:
                print(f"⚠️ AutoModelForInference failed: {e}")
                print("🔄 Falling back to AutoModelForCausalLM...")
        
        if not model_loaded:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            print("✅ AutoModelForCausalLM loaded as fallback")
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Shared generation defaults
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.max_new_tokens = 256
        self.generation_config.temperature = 0.7
        self.generation_config.do_sample = False
        self.generation_config.repetition_penalty = 1.2
        self.generation_config.no_repeat_ngram_size = 3
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config = self.generation_config

        device = next(self.model.parameters()).device
        print(f"Model loaded on {device}")

    def _get_system_prompt(self, subject: str) -> str:
        """Generate system prompt for Singapore O-Level tutoring."""
        if subject.lower() == "chemistry":
            return f"""You are an educational AI tutor for Singapore O-Level Chemistry students (ages 13-16).
You specialise in Chemistry following the MOE curriculum.

**GUIDELINES:**
1. Provide clear, step-by-step explanations for chemistry concepts
2. Use Singapore curriculum terminology and notation
3. Focus on understanding chemical formulas, reactions, and concepts
4. Maintain an encouraging, age-appropriate tone
5. Refuse off-topic or inappropriate requests politely

When explaining:
- Start with basic definitions
- Provide molecular formulas correctly (e.g., H₂O for water)
- Explain chemical properties and reactions
- Use examples relevant to daily life when helpful

Answer chemistry questions directly but explain the reasoning behind your answers.
"""
        elif subject.lower() == "physics":
            return f"""You are an educational AI tutor for Singapore O-Level Physics students (ages 13-16).
You specialise in Physics following the MOE curriculum.

**GUIDELINES:**
1. Provide clear, step-by-step explanations for physics concepts
2. Use Singapore curriculum terminology and notation
3. Focus on understanding formulas, laws, and physical principles
4. Maintain an encouraging, age-appropriate tone
5. Refuse off-topic or inappropriate requests politely

When explaining:
- Start with fundamental concepts
- Show relevant formulas and their applications
- Explain units and measurements
- Use real-world examples when helpful

Answer physics questions directly but explain the reasoning behind your answers.
"""
        else:
            return f"""You are an educational AI tutor for Singapore O-Level students (ages 13-16).
You specialise in {subject} following the MOE curriculum.

**GUIDELINES:**
1. Provide step-by-step explanations WITHOUT giving direct answers
2. Use Singapore curriculum terminology and notation
3. Highlight key concepts that appear in MOE marking schemes
4. Maintain an encouraging, age-appropriate tone
5. Refuse off-topic or inappropriate requests politely

**EXAMPLE APPROACH:**
Student: "How do I solve x² + 5x + 6 = 0?"
You: "Great question! Let's use factorization. First, we need two numbers that:
- Multiply to give 6 (the constant term)
- Add to give 5 (the coefficient of x)

Can you think of two numbers that fit these conditions?"
"""

    def generate(
        self,
        prompt: str,
        subject: str = "General",
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = 256,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None
    ) -> Dict:
        """
        Generate response from baseline model.
        
        Args:
            prompt: Student's question
            subject: Subject area
            temperature: Sampling temperature (defaults to shared generation config)
            max_new_tokens: Maximum new tokens to generate (defaults to shared generation config)
            do_sample: Whether to use sampling (defaults to shared generation config)
            repetition_penalty: Penalty for repeated token sequences (>1.0 discourages repetition)
            no_repeat_ngram_size: Prevents repeating n-grams of this size when set (>0)
            
        Returns:
            Dictionary with 'response' and 'metadata'
        """
        try:
            print(f"🔄 Starting generation for: {prompt[:50]}...")
            
            # Format prompt with system context
            system_prompt = self._get_system_prompt(subject)
            full_prompt = f"{system_prompt}\n\nStudent: {prompt}\n\nTutor:"
            print(f"📝 Full prompt length: {len(full_prompt)} chars")
            
            # Tokenize
            device = next(self.model.parameters()).device
            print(f"💻 Using device: {device}")
            
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(device)
            print(f"🎯 Input tokens: {inputs['input_ids'].shape[1]}")
            
            # Resolve generation parameters (fallback to shared config)
            resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.model.generation_config.max_new_tokens
            if resolved_max_new_tokens is None:
                resolved_max_new_tokens = 256

            resolved_temperature = temperature if temperature is not None else self.model.generation_config.temperature
            if resolved_temperature is None:
                resolved_temperature = 0.7

            resolved_do_sample = do_sample if do_sample is not None else self.model.generation_config.do_sample
            if resolved_do_sample is None:
                resolved_do_sample = False

            resolved_repetition = repetition_penalty if repetition_penalty is not None else self.model.generation_config.repetition_penalty
            if resolved_repetition is None:
                resolved_repetition = 1.2

            resolved_ngram = no_repeat_ngram_size if no_repeat_ngram_size is not None else self.model.generation_config.no_repeat_ngram_size
            if resolved_ngram is not None and resolved_ngram <= 0:
                resolved_ngram = None

            # Generate
            print(f"⚡ Starting generation with max_new_tokens={resolved_max_new_tokens}, temperature={resolved_temperature}, do_sample={resolved_do_sample}")
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": resolved_max_new_tokens,
                    "temperature": resolved_temperature,
                    "do_sample": resolved_do_sample,
                    "repetition_penalty": resolved_repetition,
                    "pad_token_id": self.model.generation_config.pad_token_id,
                    "eos_token_id": self.model.generation_config.eos_token_id,
                }

                if resolved_ngram:
                    gen_kwargs["no_repeat_ngram_size"] = resolved_ngram

                outputs = self.model.generate(**inputs, **gen_kwargs)
            print(f"✅ Generation completed! Output shape: {outputs.shape}")
            
            # Clear GPU cache after generation to prevent memory buildup
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print("🧹 MPS cache cleared")
            
            # Decode only the new tokens
            # outputs is a tensor with shape [batch_size, sequence_length]
            if outputs.numel() == 0:
                raise ValueError("Model generation returned empty output")
            
            # Ensure outputs has the expected shape [batch_size, sequence_length]
            if len(outputs.shape) < 2:
                raise ValueError(f"Unexpected output shape: {outputs.shape}. Expected [batch_size, sequence_length]")
            
            if outputs.shape[0] == 0:
                raise ValueError("Output batch size is 0")
            
            input_length = inputs['input_ids'].shape[1]
            output_length = outputs.shape[1]  # sequence_length dimension
            
            if output_length < input_length:
                raise ValueError(f"Output length ({output_length}) is less than input length ({input_length})")
            
            # Get the first batch item and extract only the newly generated tokens
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )

            neg_log_likelihood = outputs.loss
            avg_neg_log_likelihood = neg_log_likelihood / output_length
            ppl = torch.exp(avg_neg_log_likelihood).item()
            
            return {
                "response": response_text.strip(),
                "metadata": {
                    "model_name": self.model_name,
                    "prompt_tokens": input_length,
                    "generated_tokens": output_length - input_length,
                    "total_tokens": output_length,
                    "perplexity": ppl
                }
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "metadata": {"error": str(e)}
            }

    def batch_generate(
        self, 
        prompts: List[str], 
        subject: str = "General"
    ) -> List[Dict]:
        """Generate responses for multiple prompts (for evaluation)."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing {i+1}/{len(prompts)}...")
            result = self.generate(prompt, subject=subject)
            results.append(result)
        return results


if __name__ == "__main__":
    # Quick manual smoke test
    model = FineTunedModel(model_name="meta-llama/Llama-3.2-3B-Instruct")

    result = model.generate(
        prompt="Explain how the respiratory system facilitates gas exchange.",
        subject="Biology",
    )

    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")