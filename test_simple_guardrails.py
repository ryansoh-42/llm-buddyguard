#!/usr/bin/env python3
"""
Quick test script to verify the reverted guardrails are working.
"""

from src.guardrails import EducationalGuardrails

def test_guardrails():
    guardrails = EducationalGuardrails()
    
    # Test cases
    test_prompts = [
        "atom",
        "H2O", 
        "What is chemistry?",
        "Tell me about politics",
        "How do chemical bonds work?"
    ]
    
    print("Testing reverted simple guardrails:")
    print("=" * 50)
    
    for prompt in test_prompts:
        result = guardrails.apply_guardrails(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Allowed: {result['allowed']}")
        print(f"Message: {result['message']}")

if __name__ == "__main__":
    test_guardrails()