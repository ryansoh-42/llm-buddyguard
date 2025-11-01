from src.models.frontier import FrontierModel

def test_frontier_model():
    """Test frontier model with a simple question."""
    print("Testing Frontier Model (GPT-4o)...")
    print("-" * 50)
    
    try:
        model = FrontierModel()
        print("Model loaded successfully\n")
        
        result = model.generate(
            "What is Pythagoras theorem?", 
            subject="Mathematics"
        )
        
        print("RESPONSE:")
        print(result['response'])
        print(f"\nConfidence: {result['confidence']}")
        print(f"Tokens used: {result['usage']['total_tokens']}")
        print("\nTest passed!")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file")
        print("2. Added OPENAI_API_KEY to .env")
        print("3. Obtained API key from https://platform.openai.com/api-keys")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        
def test_frontier_confidence_metrics():
    """Test that frontier model returns detailed confidence metrics"""
    from src.models.frontier import FrontierModel
    
    model = FrontierModel()
    result = model.generate(
        prompt="What is 2 + 2?",
        subject="Mathematics"
    )
    
    # Check confidence_metrics structure
    assert "confidence_metrics" in result
    metrics = result["confidence_metrics"]
    
    assert "confidence" in metrics
    assert "uncertainty" in metrics
    assert "high_uncertainty_count" in metrics
    assert "avg_entropy" in metrics
    assert "total_tokens" in metrics
    
    # Check value ranges
    assert 0 <= metrics["confidence"] <= 1
    assert 0 <= metrics["uncertainty"] <= 1
    assert metrics["total_tokens"] > 0
    
    print("Confidence metrics test passed")

if __name__ == "__main__":
    test_frontier_model()