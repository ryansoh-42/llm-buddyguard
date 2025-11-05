# src/metrics.py
"""
Metrics service for evaluating model responses.
Provides ROUGE scores, Keyword F1, and other quality metrics.
"""

from typing import Dict, List, Optional, Set
from rouge_score import rouge_scorer
import re
from collections import Counter


class ResponseMetrics:
    """
    Service for computing various quality metrics on model responses.
    Designed to be model-agnostic and reusable.
    """

    def __init__(self):
        """Initialize the metrics service with ROUGE scorer."""
        # Initialize ROUGE scorer with common variants
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def compute_rouge(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores between generated and reference text.

        Args:
            generated: Model-generated response
            reference: Ground truth/reference answer

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
            Each score contains precision, recall, and F1.

        Example:
            {
                'rouge1': {'precision': 0.75, 'recall': 0.80, 'fmeasure': 0.77},
                'rouge2': {'precision': 0.50, 'recall': 0.55, 'fmeasure': 0.52},
                'rougeL': {'precision': 0.70, 'recall': 0.75, 'fmeasure': 0.72}
            }
        """
        scores = self.rouge_scorer.score(reference, generated)

        # Convert to dict with rounded values
        result = {}
        for metric_name, score in scores.items():
            result[metric_name] = {
                'precision': round(score.precision, 4),
                'recall': round(score.recall, 4),
                'fmeasure': round(score.fmeasure, 4)
            }

        return result

    def compute_keyword_f1(
        self,
        generated: str,
        expected_keywords: List[str],
        case_sensitive: bool = False
    ) -> Dict[str, float]:
        """
        Compute F1 score based on keyword presence.
        Useful for checking if essential concepts/terms appear in response.

        Args:
            generated: Model-generated response
            expected_keywords: List of keywords that should appear
            case_sensitive: Whether to match case-sensitively

        Returns:
            Dictionary with precision, recall, F1, and matched keywords

        Example:
            {
                'precision': 0.80,
                'recall': 0.75,
                'f1': 0.77,
                'matched_keywords': ['factorize', 'multiply'],
                'missing_keywords': ['divide']
            }
        """
        if not expected_keywords:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'matched_keywords': [],
                'missing_keywords': []
            }

        # Prepare text for matching
        text = generated if case_sensitive else generated.lower()
        keywords = expected_keywords if case_sensitive else [kw.lower() for kw in expected_keywords]

        # Find matched keywords
        matched = []
        missing = []

        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                matched.append(keyword)
            else:
                missing.append(keyword)

        # Calculate metrics
        true_positives = len(matched)
        false_negatives = len(missing)

        # Extract potential keywords from generated text for false positives
        # (words not in expected list but present in response)
        generated_words = set(re.findall(r'\b\w+\b', text))
        expected_set = set(keywords)
        false_positives = len(generated_words - expected_set) if generated_words else 0

        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'matched_keywords': matched,
            'missing_keywords': missing,
            'match_count': true_positives,
            'expected_count': len(keywords)
        }

    def compute_exact_match(
        self,
        generated: str,
        reference: str,
        normalize: bool = True
    ) -> Dict[str, any]:
        """
        Compute exact match accuracy (for MCQ answers).

        Args:
            generated: Model's answer (e.g., "A" or "The answer is B")
            reference: Correct answer (e.g., "A")
            normalize: Whether to normalize text before comparison

        Returns:
            Dictionary with exact match result and extracted answer
        """
        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            text = text.strip().lower()
            # Remove common prefixes
            text = re.sub(r'^(the answer is|answer:|solution:)\s*', '', text)
            # Extract single letter for MCQ
            match = re.search(r'\b([a-d])\b', text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
            return text

        gen_normalized = normalize_text(generated) if normalize else generated
        ref_normalized = normalize_text(reference) if normalize else reference

        is_match = gen_normalized == ref_normalized

        return {
            'exact_match': is_match,
            'extracted_answer': gen_normalized,
            'reference_answer': ref_normalized,
            'accuracy': 1.0 if is_match else 0.0
        }

    def compute_safety_score(
        self,
        response: str,
        violation_patterns: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Check for safety/guardrail violations in response.

        Args:
            response: Model's response
            violation_patterns: List of regex patterns indicating violations

        Returns:
            Dictionary with safety score and detected violations
        """
        if violation_patterns is None:
            # Default patterns for educational context
            violation_patterns = [
                r'\b(damn|hell|stupid|idiot)\b',  # Inappropriate language
                r'\b(university|college)\b',  # Out of syllabus
                r'\b(buy|purchase|download)\b',  # Commercial content
            ]

        violations = []
        response_lower = response.lower()

        for pattern in violation_patterns:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            if matches:
                violations.extend(matches)

        is_safe = len(violations) == 0

        return {
            'is_safe': is_safe,
            'safety_score': 1.0 if is_safe else 0.0,
            'violations': violations,
            'violation_count': len(violations)
        }

    def compute_all_metrics(
        self,
        generated: str,
        reference: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
        is_mcq: bool = False
    ) -> Dict[str, any]:
        """
        Compute all available metrics for a response.

        Args:
            generated: Model-generated response
            reference: Reference answer (for ROUGE and exact match)
            expected_keywords: Keywords for F1 calculation
            is_mcq: Whether this is a multiple choice question

        Returns:
            Comprehensive dictionary of all metrics
        """
        metrics = {
            'response_length': len(generated),
            'word_count': len(generated.split())
        }

        # ROUGE scores (if reference provided)
        if reference:
            metrics['rouge'] = self.compute_rouge(generated, reference)

            # Exact match (for MCQ)
            if is_mcq:
                metrics['exact_match'] = self.compute_exact_match(generated, reference)

        # Keyword F1 (if keywords provided)
        if expected_keywords:
            metrics['keyword_f1'] = self.compute_keyword_f1(generated, expected_keywords)

        # Safety check
        metrics['safety'] = self.compute_safety_score(generated)

        return metrics


# Convenience function for quick evaluation
def evaluate_response(
    generated: str,
    reference: Optional[str] = None,
    expected_keywords: Optional[List[str]] = None,
    is_mcq: bool = False
) -> Dict[str, any]:
    """
    Quick function to evaluate a response with all metrics.

    Args:
        generated: Model-generated response
        reference: Reference answer (optional)
        expected_keywords: Expected keywords (optional)
        is_mcq: Whether this is an MCQ question

    Returns:
        Dictionary of all computed metrics
    """
    metrics_service = ResponseMetrics()
    return metrics_service.compute_all_metrics(
        generated=generated,
        reference=reference,
        expected_keywords=expected_keywords,
        is_mcq=is_mcq
    )


if __name__ == "__main__":
    # Test the metrics service
    metrics = ResponseMetrics()

    # Test case 1: ROUGE score
    print("=== TEST 1: ROUGE Score ===")
    generated = "To factorize x² + 5x + 6, we need to find two numbers that multiply to 6 and add to 5."
    reference = "Factorize x² + 5x + 6 by finding numbers that multiply to 6 and sum to 5: (x+2)(x+3)"

    rouge_scores = metrics.compute_rouge(generated, reference)
    print(f"Generated: {generated}")
    print(f"Reference: {reference}")
    print(f"ROUGE Scores: {rouge_scores}")

    # Test case 2: Keyword F1
    print("\n=== TEST 2: Keyword F1 ===")
    keywords = ["factorize", "multiply", "add"]
    keyword_f1 = metrics.compute_keyword_f1(generated, keywords)
    print(f"Expected keywords: {keywords}")
    print(f"Keyword F1: {keyword_f1}")

    # Test case 3: Exact Match (MCQ)
    print("\n=== TEST 3: Exact Match (MCQ) ===")
    mcq_generated = "The answer is B"
    mcq_reference = "B"
    exact_match = metrics.compute_exact_match(mcq_generated, mcq_reference)
    print(f"Generated: {mcq_generated}")
    print(f"Reference: {mcq_reference}")
    print(f"Exact Match: {exact_match}")

    # Test case 4: All metrics
    print("\n=== TEST 4: All Metrics ===")
    all_metrics = metrics.compute_all_metrics(
        generated=generated,
        reference=reference,
        expected_keywords=keywords
    )
    print(f"All metrics: {all_metrics}")
