# src/evaluation.py
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import re

class ModelEvaluator:
    """
    Evaluates model performance on educational tutoring tasks.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_relevance(self, response: str, expected_keywords: List[str]) -> float:
        """
        Check if response contains expected educational keywords.
        
        Args:
            response: Model's response
            expected_keywords: Keywords that should appear (e.g., ["factorize", "multiply", "add"])
            
        Returns:
            Relevance score (0-1)
        """
        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        return matches / len(expected_keywords) if expected_keywords else 0.0
    
    def check_no_direct_answer(self, response: str) -> bool:
        """
        Verify that response doesn't give direct answer.
        Looks for phrases like "the answer is", "x = ", etc.
        
        Args:
            response: Model's response
            
        Returns:
            True if no direct answer detected, False otherwise
        """
        direct_answer_patterns = [
            r"the answer is",
            r"the solution is",
            r"x\s*=\s*\d+",
            r"answer:\s*\d+",
            r"result:\s*\d+"
        ]
        
        response_lower = response.lower()
        for pattern in direct_answer_patterns:
            if re.search(pattern, response_lower):
                return False
        return True
    
    def evaluate_step_by_step(self, response: str) -> float:
        """
        Score response based on step-by-step guidance indicators.
        
        Args:
            response: Model's response
            
        Returns:
            Score (0-1) based on step-by-step indicators
        """
        step_indicators = [
            "first", "second", "third", "next", "then",
            "step 1", "step 2", "step 3",
            "let's", "can you", "try to"
        ]
        
        response_lower = response.lower()
        matches = sum(1 for indicator in step_indicators if indicator in response_lower)
        
        # Normalize by expected number of steps (assume 3-5 is good)
        return min(matches / 3.0, 1.0)
    
    def evaluate_tone(self, response: str) -> float:
        """
        Score response based on encouraging, age-appropriate tone.
        
        Args:
            response: Model's response
            
        Returns:
            Tone score (0-1)
        """
        encouraging_phrases = [
            "great question", "good thinking", "you're on the right track",
            "well done", "excellent", "that's correct", "nice work"
        ]
        
        response_lower = response.lower()
        has_encouragement = any(phrase in response_lower for phrase in encouraging_phrases)
        
        # Check for overly complex language (red flag)
        avg_word_length = sum(len(word) for word in response.split()) / max(len(response.split()), 1)
        complexity_penalty = 1.0 if avg_word_length < 7 else 0.8
        
        return (1.0 if has_encouragement else 0.7) * complexity_penalty
    
    def evaluate_confidence(self, confidence_metrics: Dict) -> Dict:
        """
        Evaluate model confidence and uncertainty.
        
        Args:
            confidence_metrics: Dict from model's confidence calculation
        
        Returns:
            Confidence evaluation scores
        """
        if not confidence_metrics or confidence_metrics.get("confidence", 0) == 0:
            return {
                "confidence_score": 0.0,
                "raw_confidence": 0.0,
                "uncertainty": 0.0,
                "avg_entropy": 0.0,
                "uncertain_token_ratio": 0.0,
                "needs_review": True,
                "review_reason": "No confidence data"
            }
        
        confidence = confidence_metrics.get("confidence", 0)
        uncertainty = confidence_metrics.get("uncertainty", 0)
        entropy = confidence_metrics.get("avg_entropy", 0)
        high_uncertain_tokens = confidence_metrics.get("high_uncertainty_count", 0)
        total_tokens = confidence_metrics.get("total_tokens", 1)
        
        # Calculate metrics
        uncertain_token_ratio = high_uncertain_tokens / total_tokens if total_tokens > 0 else 0
        
        # Determine if response needs review
        needs_review = False
        review_reasons = []
        
        if confidence < 0.6:
            needs_review = True
            review_reasons.append(f"Low confidence ({confidence:.2f})")
        
        if uncertainty > 0.4:
            needs_review = True
            review_reasons.append(f"High avg uncertainty ({uncertainty:.2f})")
        
        if uncertain_token_ratio > 0.2:  # More than 20% of tokens uncertain
            needs_review = True
            review_reasons.append(f"Many uncertain tokens ({uncertain_token_ratio:.1%})")
        
        # Confidence score for overall evaluation (0-1)
        # Penalize both low confidence and high uncertainty
        confidence_score = confidence * (1 - uncertainty * 0.5)
        
        return {
            "confidence_score": round(confidence_score, 4),
            "raw_confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "avg_entropy": round(entropy, 4),
            "uncertain_token_ratio": round(uncertain_token_ratio, 4),
            "needs_review": needs_review,
            "review_reason": "; ".join(review_reasons) if review_reasons else "OK"
        }
    
    def evaluate_response(
        self, 
        response: str, 
        confidence_metrics: Dict = None,
        expected_keywords: List[str] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single response.
        
        Args:
            response: Model's response
            expected_keywords: Optional keywords for relevance check
            
        Returns:
            Dictionary of metric scores
        """
        results = {
            "no_direct_answer": 1.0 if self.check_no_direct_answer(response) else 0.0,
            "step_by_step_score": self.evaluate_step_by_step(response),
            "tone_score": self.evaluate_tone(response),
            "response_length": len(response),
            "word_count": len(response.split())
        }
        
        if expected_keywords:
            results["relevance_score"] = self.evaluate_relevance(response, expected_keywords)
        
        base_scores = [
        results["no_direct_answer"],
        results["step_by_step_score"],
        results["tone_score"]
        ]
        
        if confidence_metrics:
            base_scores.append(results["confidence_score"])
        
            results["overall_score"] = sum(base_scores) / len(base_scores)
            
            return results
    
    def evaluate_dataset(
        self, 
        responses: List[Dict], 
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple responses and generate report.
        
        Args:
            responses: List of dicts with 'prompt', 'response', 'expected_keywords'
            save_path: Optional path to save results CSV
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for item in responses:
            evaluation = self.evaluate_response(
                item['response'],
                item.get('expected_keywords')
            )
            evaluation['prompt'] = item['prompt']
            evaluation['response'] = item['response']
            results.append(evaluation)
        
        df = pd.DataFrame(results)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        # Print summary statistics
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total responses evaluated: {len(df)}")
        print(f"Average overall score: {df['overall_score'].mean():.2f}")
        print(f"No direct answer rate: {df['no_direct_answer'].mean():.2%}")
        print(f"Avg step-by-step score: {df['step_by_step_score'].mean():.2f}")
        print(f"Avg tone score: {df['tone_score'].mean():.2f}")

        # Add confidence metrics if available
        if 'raw_confidence' in df.columns:
            print(f"\n=== CONFIDENCE METRICS ===")
            print(f"Avg confidence: {df['raw_confidence'].mean():.2f}")
            print(f"Avg uncertainty: {df['uncertainty'].mean():.2f}")
            print(f"Avg entropy: {df['avg_entropy'].mean():.2f}")
            print(f"Responses needing review: {df['needs_review'].sum()}/{len(df)}")
        
        return df


if __name__ == "__main__":
    # Test evaluator
    evaluator = ModelEvaluator()
    
    # Example response
    test_response = """Great question! Let's solve this step by step.
    First, we need to factorize the quadratic expression.
    Can you identify two numbers that multiply to 6 and add to 5?"""
    
    results = evaluator.evaluate_response(
        test_response,
        expected_keywords=["factorize", "multiply", "add"]
    )
    
    print(results)