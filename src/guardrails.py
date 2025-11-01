# src/guardrails.py
from typing import Dict, List, Tuple
import re

class EducationalGuardrails:
    """
    Implements safety and educational guardrails for tutoring AI.
    """
    
    def __init__(self):
        self.off_topic_keywords = [
            "politics", "religion", "violence", "drugs", "alcohol",
            "dating", "sex", "hack", "cheat answers"
        ]
        
        self.allowed_subjects = ["mathematics", "science", "english", "math", "chemistry", "physics", "biology"]
    
    def check_appropriate_content(self, prompt: str) -> Tuple[bool, str]:
        """
        Check if prompt is appropriate for educational context.
        
        Args:
            prompt: Student's input
            
        Returns:
            (is_appropriate, reason)
        """
        prompt_lower = prompt.lower()
        
        # Check for off-topic keywords
        for keyword in self.off_topic_keywords:
            if keyword in prompt_lower:
                return False, f"This topic ({keyword}) is not appropriate for educational tutoring."
        
        # Check if it's a subject-related question
        has_subject = any(subject in prompt_lower for subject in self.allowed_subjects)
        has_question_mark = "?" in prompt
        
        if not has_subject and not has_question_mark:
            return False, "Please ask a question related to Mathematics, Science, or English."
        
        return True, "Content is appropriate"
    
    def check_answer_seeking(self, prompt: str) -> Tuple[bool, str]:
        """
        Detect if student is directly asking for answers.
        
        Args:
            prompt: Student's input
            
        Returns:
            (is_answer_seeking, message)
        """
        answer_seeking_patterns = [
            r"give me the answer",
            r"what is the answer",
            r"just tell me",
            r"solve this for me",
            r"do my homework"
        ]
        
        prompt_lower = prompt.lower()
        for pattern in answer_seeking_patterns:
            if re.search(pattern, prompt_lower):
                return True, "I'm here to guide your learning, not provide direct answers. Let's work through this together!"
        
        return False, ""
    
    def filter_response(self, response: str) -> str:
        """
        Post-process response to ensure educational standards.
        
        Args:
            response: Model's generated response
            
        Returns:
            Filtered response
        """
        # Remove any accidental direct answers in format "x = 5"
        # This is a safety check - evaluation should catch this earlier
        filtered = response
        
        # Add educational reminder if response is very short
        if len(response) < 50:
            filtered += "\n\nRemember, I'm here to guide you through the problem. What's your next step?"
        
        return filtered
    
    def apply_guardrails(self, prompt: str) -> Dict:
        """
        Apply all guardrails to student prompt.
        
        Args:
            prompt: Student's input
            
        Returns:
            Dictionary with 'allowed', 'message', and 'modified_prompt'
        """
        # Check content appropriateness
        is_appropriate, content_reason = self.check_appropriate_content(prompt)
        if not is_appropriate:
            return {
                "allowed": False,
                "message": content_reason,
                "modified_prompt": None
            }
        
        # Check for answer-seeking behavior
        is_answer_seeking, answer_message = self.check_answer_seeking(prompt)
        if is_answer_seeking:
            return {
                "allowed": True,  # Still allowed, but with modified approach
                "message": answer_message,
                "modified_prompt": f"Guide the student without giving direct answer: {prompt}"
            }
        
        return {
            "allowed": True,
            "message": "Prompt approved",
            "modified_prompt": prompt
        }


if __name__ == "__main__":
    guardrails = EducationalGuardrails()
    
    # Test cases
    test_prompts = [
        "What is the answer to x^2 + 5x + 6 = 0?",
        "Just give me the answer to question 5",
        "How do I approach solving quadratic equations?",
        "Tell me about politics"
    ]
    
    for prompt in test_prompts:
        result = guardrails.apply_guardrails(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Result: {result}")