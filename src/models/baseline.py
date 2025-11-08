# src/models/baseline.py
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import AutoModelForInference
    HAS_INFERENCE_MODEL = True
    print("✅ AutoModelForInference available")
except ImportError:
    HAS_INFERENCE_MODEL = False
    print("ℹ️ AutoModelForInference not available, using AutoModelForCausalLM")
import torch


class BaselineModel:
    """
    Open-weight baseline model (no fine-tuning) for O-Level tutoring.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto",
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
                    low_cpu_mem_usage=True,
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
                low_cpu_mem_usage=True,
            )
            print("✅ AutoModelForCausalLM loaded as fallback")

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_obj = next(self.model.parameters()).device
        print(f"Model loaded on {device_obj}")

    def _get_system_prompt(self, subject: str) -> str:
        """Generate system prompt for Singapore O-Level tutoring."""
        if subject.lower() == "chemistry":
            return (
                "You are an educational AI tutor for Singapore O-Level Chemistry students (ages 13-16).\n"
                "You specialise in Chemistry following the MOE curriculum.\n\n"
                "**GUIDELINES:**\n"
                "1. Provide clear, step-by-step explanations for chemistry concepts\n"
                "2. Use Singapore curriculum terminology and notation\n"
                "3. Focus on understanding chemical formulas, reactions, and concepts\n"
                "4. Maintain an encouraging, age-appropriate tone\n"
                "5. Refuse off-topic or inappropriate requests politely\n\n"
                "When explaining:\n"
                "- Start with basic definitions\n"
                "- Provide molecular formulas correctly (e.g., H₂O for water)\n"
                "- Explain chemical properties and reactions\n"
                "- Use examples relevant to daily life when helpful\n\n"
                "Answer chemistry questions directly but explain the reasoning behind your answers.\n"
            )
        if subject.lower() == "physics":
            return (
                "You are an educational AI tutor for Singapore O-Level Physics students (ages 13-16).\n"
                "You specialise in Physics following the MOE curriculum.\n\n"
                "**GUIDELINES:**\n"
                "1. Provide clear, step-by-step explanations for physics concepts\n"
                "2. Use Singapore curriculum terminology and notation\n"
                "3. Focus on understanding formulas, laws, and physical principles\n"
                "4. Maintain an encouraging, age-appropriate tone\n"
                "5. Refuse off-topic or inappropriate requests politely\n\n"
                "When explaining:\n"
                "- Start with fundamental concepts\n"
                "- Show relevant formulas and their applications\n"
                "- Explain units and measurements\n"
                "- Use real-world examples when helpful\n\n"
                "Answer physics questions directly but explain the reasoning behind your answers.\n"
            )

        return (
            "You are an educational AI tutor for Singapore O-Level students (ages 13-16).\n"
            f"You specialise in {subject} following the MOE curriculum.\n\n"
            "**GUIDELINES:**\n"
            "1. Provide step-by-step explanations WITHOUT giving direct answers\n"
            "2. Use Singapore curriculum terminology and notation\n"
            "3. Highlight key concepts that appear in MOE marking schemes\n"
            "4. Maintain an encouraging, age-appropriate tone\n"
            "5. Refuse off-topic or inappropriate requests politely\n\n"
            "**EXAMPLE APPROACH:**\n"
            'Student: "How do I solve x² + 5x + 6 = 0?"\n'
            'You: "Great question! Let\'s use factorization. First, we need two numbers that:\n'
            "- Multiply to give 6 (the constant term)\n"
            "- Add to give 5 (the coefficient of x)\n\n"
            "Can you think of two numbers that fit these conditions?\"\n"
        )

    def generate(
        self,
        prompt: str,
        subject: str = "Mathematics",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        do_sample: bool = True,
    ) -> Dict:
        """
        Generate response from baseline model.

        Args:
            prompt: Student's question
            subject: Subject area
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling

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
            device_obj = next(self.model.parameters()).device
            print(f"💻 Using device: {device_obj}")

            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device_obj)
            print(f"🎯 Input tokens: {inputs['input_ids'].shape[1]}")

            # Generate
            print(
                "⚡ Starting generation with "
                f"max_new_tokens={max_new_tokens}, temperature={temperature}"
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            print(f"✅ Generation completed! Output shape: {outputs.shape}")

            # Clear GPU/MPS cache after generation to prevent memory buildup
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print("🧹 MPS cache cleared")

            if outputs.numel() == 0:
                raise ValueError("Model generation returned empty output")

            if len(outputs.shape) < 2:
                raise ValueError(
                    f"Unexpected output shape: {outputs.shape}. "
                    "Expected [batch_size, sequence_length]"
                )

            if outputs.shape[0] == 0:
                raise ValueError("Output batch size is 0")

            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.shape[1]

            if output_length < input_length:
                raise ValueError(
                    f"Output length ({output_length}) is less than input length ({input_length})"
                )

            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            return {
                "response": response_text.strip(),
                "metadata": {
                    "model_name": self.model_name,
                    "prompt_tokens": input_length,
                    "generated_tokens": output_length - input_length,
                    "total_tokens": output_length,
                },
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "metadata": {"error": str(e)},
            }

    def batch_generate(
        self,
        prompts: List[str],
        subject: str = "Mathematics",
    ) -> List[Dict]:
        """Generate responses for multiple prompts (for evaluation)."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing {i+1}/{len(prompts)}...")
            result = self.generate(prompt, subject=subject)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test baseline model
    model = BaselineModel(model_name="meta-llama/Llama-3.2-3B-Instruct")

    result = model.generate(
        prompt="How do I find the area of a circle?",
        subject="Mathematics",
    )

    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")

