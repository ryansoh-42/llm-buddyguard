# src/metrics.py
"""
Metrics service for evaluating model responses.
Provides ROUGE scores, Keyword F1, and other quality metrics.
"""

from typing import Dict, List, Optional, Set, Tuple
from rouge_score import rouge_scorer
import re
from collections import Counter
import Levenshtein

# Optional spaCy import (lazy loading to avoid numpy issues)
_nlp = None

def _get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise ImportError(
                f"spaCy model not available: {e}. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


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
        normalize: bool = True,
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Compute exact match accuracy (for MCQ answers).
        Optionally checks for keyword presence in the response.

        Args:
            generated: Model's answer (e.g., "A" or "The answer is B")
            reference: Correct answer (e.g., "A")
            normalize: Whether to normalize text before comparison
            expected_keywords: Optional keywords to check for in response

        Returns:
            Dictionary with exact match result, extracted answer, and keyword analysis
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

        result = {
            'exact_match': is_match,
            'extracted_answer': gen_normalized,
            'reference_answer': ref_normalized,
            'accuracy': 1.0 if is_match else 0.0
        }

        # Add keyword analysis if provided
        if expected_keywords:
            keyword_result = self.compute_keyword_f1(generated, expected_keywords)
            result['keyword_f1'] = keyword_result['f1']
            result['keyword_recall'] = keyword_result['recall']
            result['matched_keywords'] = keyword_result['matched_keywords']
            result['missing_keywords'] = keyword_result['missing_keywords']

        return result

    def extract_concepts_from_text(
        self,
        text: str,
        use_spacy: bool = True
    ) -> List[str]:
        """
        Dynamically extract key concepts (nouns and verbs) from text.
        NO hardcoded lists - adapts to any subject/question.

        Args:
            text: Text to extract concepts from
            use_spacy: Use spaCy NLP (if False, falls back to simple word extraction)

        Returns:
            List of concepts in order they appear in text

        Examples:
            "First, prophase occurs. Then metaphase aligns chromosomes."
            -> ["prophase", "occur", "metaphase", "align", "chromosome"]

            "Factorize the expression, then multiply the terms."
            -> ["factorize", "expression", "multiply", "term"]
        """
        if use_spacy:
            try:
                nlp = _get_nlp()
                doc = nlp(text)

                concepts = []

                # Extract noun chunks (handles multi-word terms like "cell division")
                for chunk in doc.noun_chunks:
                    # Only keep nouns (not pronouns like "it", "this")
                    if chunk.root.pos_ in ["NOUN", "PROPN"]:
                        # Use root word for consistency
                        concepts.append((chunk.start_char, chunk.root.lemma_.lower()))

                # Extract main verbs (not auxiliaries like "is", "have", "be")
                AUXILIARY_VERBS = {"be", "have", "do", "will", "would", "should", "could", "may", "might", "must", "can"}
                for token in doc:
                    if token.pos_ == "VERB" and token.lemma_.lower() not in AUXILIARY_VERBS:
                        # Only include important verbs (ROOT or clause verbs)
                        if token.dep_ in ["ROOT", "advcl", "xcomp", "ccomp"]:
                            concepts.append((token.idx, token.lemma_.lower()))

                # Sort by position in text and remove duplicates (keep first occurrence)
                concepts.sort(key=lambda x: x[0])
                seen = set()
                unique_concepts = []
                for pos, concept in concepts:
                    if concept not in seen:
                        seen.add(concept)
                        unique_concepts.append(concept)

                return unique_concepts

            except ImportError:
                # Fall back to simple extraction
                use_spacy = False

        if not use_spacy:
            # Fallback: Extract words that are likely concepts (nouns/verbs)
            # Filter out common filler words and step markers
            FILLER_WORDS = {
                "first", "second", "third", "then", "next", "finally", "after", "before",
                "during", "while", "when", "where", "what", "which", "that", "this",
                "these", "those", "your", "their", "from", "into", "with", "about",
                "also", "will", "would", "should", "could", "must", "need", "followed"
            }

            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            # Remove duplicates and filler words while preserving order
            seen = set()
            return [w for w in words if w not in FILLER_WORDS and not (w in seen or seen.add(w))]

    def compute_order_score(
        self,
        generated: str,
        reference: str,
        use_spacy: bool = True
    ) -> Dict[str, any]:
        """
        Evaluate whether concepts appear in correct order.
        Dynamically extracts concepts from reference, then finds them in generated text.

        Args:
            generated: Model-generated response
            reference: Reference response with correct concept order
            use_spacy: Use spaCy for extraction (default: True)

        Returns:
            Dictionary with order correctness metrics

        Example:
            reference = "First prophase occurs, then metaphase, then anaphase"
            generated = "First metaphase happens, then prophase, then anaphase"

            Result: {
                'order_score': 0.33,
                'reference_order': ['prophase', 'occur', 'metaphase', 'anaphase'],
                'generated_order': ['metaphase', 'happen', 'prophase', 'anaphase'],
                'matched_concepts': ['metaphase', 'prophase', 'anaphase'],
                'edit_distance': 2,
                'correct_order': False
            }
        """
        # Step 1: Extract concepts from reference (defines what we're looking for)
        ref_concepts = self.extract_concepts_from_text(reference, use_spacy=use_spacy)

        # Step 2: Extract concepts from generated
        gen_concepts = self.extract_concepts_from_text(generated, use_spacy=use_spacy)

        # Step 3: Find which reference concepts appear in generated (and in what order)
        matched_in_generated = []
        for concept in ref_concepts:
            if concept in gen_concepts:
                matched_in_generated.append(concept)

        # Handle edge cases
        if not ref_concepts:
            return {
                'order_score': 0.0,
                'reference_order': [],
                'generated_order': gen_concepts,
                'matched_concepts': [],
                'missing_concepts': [],
                'edit_distance': 0,
                'correct_order': False,
                'normalized_distance': 1.0,
                'message': 'No concepts found in reference'
            }

        if not matched_in_generated:
            return {
                'order_score': 0.0,
                'reference_order': ref_concepts,
                'generated_order': gen_concepts,
                'matched_concepts': [],
                'missing_concepts': ref_concepts,
                'edit_distance': len(ref_concepts),
                'correct_order': False,
                'normalized_distance': 1.0,
                'message': 'No reference concepts found in generated response'
            }

        # Compute Levenshtein edit distance on matched concepts only
        edit_dist = Levenshtein.distance(ref_concepts, matched_in_generated)

        # Normalize by reference length (what we expected)
        normalized_dist = edit_dist / len(ref_concepts) if len(ref_concepts) > 0 else 0.0

        # Order score: 1.0 = perfect match, 0.0 = completely different
        order_score = 1.0 - normalized_dist

        # Check if order is exactly correct
        correct_order = (ref_concepts == matched_in_generated)

        # Find missing concepts
        missing_concepts = [c for c in ref_concepts if c not in matched_in_generated]

        return {
            'order_score': round(order_score, 4),
            'reference_order': ref_concepts,
            'generated_order': gen_concepts,
            'matched_concepts': matched_in_generated,
            'missing_concepts': missing_concepts,
            'edit_distance': edit_dist,
            'correct_order': correct_order,
            'normalized_distance': round(normalized_dist, 4),
            'message': 'Order matches' if correct_order else f'{edit_dist} edit(s) needed for correct order'
        }

    def compute_all_metrics(
        self,
        generated: str,
        reference: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
        is_mcq: bool = False,
        check_order: bool = False
    ) -> Dict[str, any]:
        """
        Compute all available metrics for a response.

        Args:
            generated: Model-generated response
            reference: Reference answer (for ROUGE and exact match)
            expected_keywords: Keywords for F1 calculation
            is_mcq: Whether this is a multiple choice question
            check_order: Whether to evaluate reasoning step order (requires reference)

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

            # Exact match (for MCQ) - includes keyword analysis if provided
            if is_mcq:
                metrics['exact_match'] = self.compute_exact_match(
                    generated,
                    reference,
                    expected_keywords=expected_keywords
                )

            # Order scoring (if requested)
            if check_order:
                metrics['order'] = self.compute_order_score(generated, reference)

        # Keyword F1 (if keywords provided)
        if expected_keywords:
            metrics['keyword_f1'] = self.compute_keyword_f1(generated, expected_keywords)

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

    # Test case 4: Order scoring
    print("\n=== TEST 4: Order Scoring ===")
    ref_order = "First, factorize the expression. Then multiply the factors. Finally, simplify the result."
    gen_correct = "First, factorize the expression. Then multiply the factors. Finally, simplify the result."
    gen_wrong = "First, multiply the factors. Then factorize the expression. Finally, simplify the result."

    print("Correct order:")
    order_correct = metrics.compute_order_score(gen_correct, ref_order, use_spacy=False)
    print(f"  Reference: {order_correct['reference_order']}")
    print(f"  Generated: {order_correct['generated_order']}")
    print(f"  Order Score: {order_correct['order_score']} - {order_correct['message']}")

    print("\nWrong order:")
    order_wrong = metrics.compute_order_score(gen_wrong, ref_order, use_spacy=False)
    print(f"  Reference: {order_wrong['reference_order']}")
    print(f"  Generated: {order_wrong['generated_order']}")
    print(f"  Order Score: {order_wrong['order_score']} - {order_wrong['message']}")

    # Test case 5: All metrics
    print("\n=== TEST 5: All Metrics ===")
    all_metrics = metrics.compute_all_metrics(
        generated=generated,
        reference=reference,
        expected_keywords=keywords,
        check_order=False  # Set to True to include order checking
    )
    print(f"All metrics keys: {list(all_metrics.keys())}")
