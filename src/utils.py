# src/utils.py
import json
import os
from datetime import datetime
from typing import Dict, Any
import torch

def get_device() -> str:
    """Get optimal device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def save_evaluation_results(results: Dict[str, Any], filename: str = None):
    """Save evaluation results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")
    return filepath

def load_evaluation_dataset(filepath: str) -> list:
    """Load evaluation dataset from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_model_comparison(baseline_result: Dict, frontier_result: Dict) -> str:
    """Format side-by-side model comparison."""
    comparison = f"""
    === MODEL COMPARISON ===
    
    BASELINE MODEL:
    Response: {baseline_result['response'][:200]}...
    Tokens: {baseline_result['metadata']['total_tokens']}
    
    FRONTIER MODEL:
    Response: {frontier_result['response'][:200]}...
    Tokens: {frontier_result['usage']['total_tokens']}
    Confidence: {frontier_result['confidence']}
    """
    return comparison