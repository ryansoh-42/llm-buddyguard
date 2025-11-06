# src/models/finetuned.py
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

class FineTunedModel:
    """
    Fine-tuned model loader for subject-specific tutoring (Physics, Chemistry, Biology).
    Loads the appropriate fine-tuned model from HuggingFace based on the subject.
    """
    
    # Mapping of subject names to HuggingFace model IDs
    MODEL_MAP = {
        "physics": "Fawl/is469_project_physics",
        "chemistry": "Fawl/is469_project_chem",
        "biology": "Fawl/is469_project_bio"
    }
    
    def __init__(
        self, 
        subject: str = "physics",
        device: str = "auto"
    ):
        """
        Initialize fine-tuned model from HuggingFace.
        
        Args:
            subject: One of "physics", "chemistry", or "biology" (case-insensitive)
            device: Device placement ("auto", "cuda", "cpu")
        """
        subject_lower = subject.lower()
        
        if subject_lower not in self.MODEL_MAP:
            raise ValueError(
                f"Subject must be one of {list(self.MODEL_MAP.keys())}, got: {subject}"
            )
        
        self.subject = subject_lower
        self.model_name = self.MODEL_MAP[subject_lower]
        
        print(f"Loading fine-tuned model for {subject}: {self.model_name}")
        
        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            token=hf_token
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        device = next(self.model.parameters()).device
        print(f"Fine-tuned {subject} model loaded on {device}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        do_sample: bool = True
    ) -> Dict:
        """
        Generate response from fine-tuned model.
        
        Args:
            prompt: Student's question
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary with 'response' and 'metadata'
        """
        try:
            # Format prompt using chat template (similar to training)
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                formatted_prompt, 
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
            if outputs.numel() == 0:
                raise ValueError("Model generation returned empty output")
            
            if len(outputs.shape) < 2:
                raise ValueError(f"Unexpected output shape: {outputs.shape}. Expected [batch_size, sequence_length]")
            
            if outputs.shape[0] == 0:
                raise ValueError("Output batch size is 0")
            
            input_length = inputs['input_ids'].shape[1]
            output_length = outputs.shape[1]
            
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
                    "subject": self.subject,
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


if __name__ == "__main__":
    # Test fine-tuned model
    import sys
    
    subject = sys.argv[1] if len(sys.argv) > 1 else "physics"
    
    model = FineTunedModel(subject=subject)
    
    result = model.generate(
        prompt="What is the relationship between force and acceleration?"
    )
    
    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")

