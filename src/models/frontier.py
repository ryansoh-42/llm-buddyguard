from dotenv import load_dotenv
load_dotenv()

import os
import time
from typing import List, Dict, Optional
from openai import OpenAI


class FrontierModel:
    """
    Wrapper for GPT-4o API with O-level tutoring capabilities.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the GPT-4o client.
        
        Args:
            model_name: Model identifier (default: "gpt-4o")
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _get_system_prompt(self, subject: str) -> str:
        """
        Generate system prompt for Singapore O-Level tutoring.
        
        Args:
            subject: One of "Mathematics", "Science", or "English"
        
        Returns:
            System prompt string
        """
        return f"""You are an AI tutor for Singapore O-Level {subject}. 
        Provide clear, step-by-step explanations appropriate for students aged 13-16."""

    def generate(
            self, 
            prompt: str, 
            subject: str,
            temperature: float = 0.7,
            max_tokens: int = 1024,
            use_logprobs: bool = True
        ) -> Dict:
        """
        Generate response from GPT-4o with confidence scoring.
        
        Args:
            prompt: Student's question
            subject: Subject area
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response length
            use_logprobs: Whether to return log probabilities for confidence
        
        Returns:
            Dictionary with 'response', 'confidence', and 'usage'
        """
        
        valid_subjects = ["Mathematics", "Science", "English"]
        if subject not in valid_subjects:
            raise ValueError(f"Subject must be one of {valid_subjects}, got: {subject}")
        
        try:
            # Create chat completion request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(subject)},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=use_logprobs,  # Enable confidence scoring
                top_logprobs=5 if use_logprobs else None  # Get top 5 alternatives
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Calculate confidence from logprobs
            confidence = self._calculate_confidence(response.choices[0].logprobs) if use_logprobs else None
            
            return {
                "response": response_text,
                "confidence": confidence,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "confidence_metrics": {
                    "confidence": 0.0,
                    "uncertainty": 0.0,
                    "high_uncertainty_count": 0,
                    "avg_entropy": 0.0,
                    "total_tokens": 0
                },
                "usage": None
            }

    def _calculate_confidence(self, logprobs_data) -> Dict:
        """
        Calculate confidence metrics from log probabilities.
        
        Args:
            logprobs_data: Logprobs object from API response
        
        Returns:
            Dictionary with confidence, uncertainty, and analysis metrics
        """
        if not logprobs_data or not logprobs_data.content:
            return {
                "confidence": 0.0,
                "uncertainty": 0.0,
                "high_uncertainty_count": 0,
                "avg_entropy": 0.0,
                "total_tokens": 0
            }
        
        import math
        
        token_probs = []
        uncertainties = []
        entropies = []
        high_uncertainty_count = 0
        
        for token_data in logprobs_data.content:
            # Get chosen token probability
            chosen_prob = math.exp(token_data.logprob)
            token_probs.append(chosen_prob)
            
            # Calculate uncertainty if alternatives are available
            if token_data.top_logprobs and len(token_data.top_logprobs) > 0:
                # Get second-best alternative probability
                second_best_prob = math.exp(token_data.top_logprobs[0].logprob)
                
                # Uncertainty = how close is the alternative?
                # 1.0 = equally likely, 0.0 = alternative very unlikely
                uncertainty_ratio = second_best_prob / chosen_prob if chosen_prob > 0 else 0
                uncertainties.append(uncertainty_ratio)
                
                # Flag if uncertainty is high (alternatives are close)
                if uncertainty_ratio > 0.5:  # Second best has >50% of top probability
                    high_uncertainty_count += 1
                
                # Calculate entropy across all top alternatives
                # Higher entropy = more spread out probability = more uncertain
                all_probs = [chosen_prob] + [math.exp(alt.logprob) for alt in token_data.top_logprobs]
                # Normalize probabilities
                prob_sum = sum(all_probs)
                norm_probs = [p / prob_sum for p in all_probs]
                # Calculate Shannon entropy
                entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in norm_probs)
                entropies.append(entropy)
        
        # Calculate aggregate metrics
        avg_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
        avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        
        return {
            "confidence": round(avg_confidence, 4),          # How confident overall
            "uncertainty": round(avg_uncertainty, 4),        # How often were alternatives close
            "high_uncertainty_count": high_uncertainty_count, # Number of "uncertain" tokens
            "avg_entropy": round(avg_entropy, 4),            # Information-theoretic uncertainty
            "total_tokens": len(token_probs)
        }

    def stream_generate(self, prompt: str, subject: str):
        """
        Stream response token-by-token (for Streamlit chat interface).
        
        Args:
            prompt: Student's question
            subject: Subject area (required: "Mathematics", "Science", or "English")
        
        Yields:
            Individual tokens as they arrive
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(subject)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True  # Enable streaming
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {e}"