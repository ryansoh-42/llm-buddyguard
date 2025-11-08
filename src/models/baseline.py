# src/models/baseline.py
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BaselineModel:
    """ 
    Open-weight baseline model (no fine-tuning) for O-Level tutoring.
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        device = next(self.model.parameters()).device
        print(f"Model loaded on {device}")

    def _get_system_prompt(self, subject: str) -> str:
        """Generate system prompt for Singapore O-Level tutoring."""
        return f"""You are an educational AI tutor for Singapore O-Level students (ages 13-16).
        You specialise in {subject} following the MOE curriculum.

        **GUIDELINES:**
        1. Provide a clear, step-by-step worked solution that mirrors the official marking scheme.
        2. Use Singapore curriculum terminology and notation.
        3. Highlight the key concepts or steps that earn method marks.
        4. Maintain an encouraging, age-appropriate tone.
        5. Always conclude with a single line in the format `Answer: <concise answer>` that matches the answer key exactly.
        6. Refuse off-topic or inappropriate requests politely.

        **EXAMPLE APPROACH:**
        Student: "How do I solve x² + 5x + 6 = 0?"
        You: "Great question! Let's solve this together.
        Step 1: Factor the quadratic into (x + a)(x + b) such that a·b = 6 and a + b = 5.
        Step 2: The pair (2, 3) works because 2·3 = 6 and 2 + 3 = 5.
        Step 3: Set each factor to zero: x + 2 = 0 or x + 3 = 0.
        Step 4: Solve to get x = -2 or x = -3.
        Answer: x = -2 or x = -3"
        """

    def generate(
        self,
        prompt: str,
        subject: str = "Mathematics",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        do_sample: bool = False
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
            # Format prompt with system context
            system_prompt = self._get_system_prompt(subject)
            full_prompt = f"{system_prompt}\n\nStudent: {prompt}\n\nTutor:"
            
            # Tokenize
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
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
            
            return {
                "response": response_text.strip(),
                "metadata": {
                    "model_name": self.model_name,
                    "prompt_tokens": input_length,
                    "generated_tokens": output_length - input_length,
                    "total_tokens": output_length
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
        subject: str = "Mathematics"
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
        subject="Mathematics"
    )
    
    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")