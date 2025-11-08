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

    def _fact_normalize_context(self, ctx):
        """
        Accept a single string or a list/tuple of strings and return List[str].
        """
        if ctx is None:
            return []
        if isinstance(ctx, str):
            return [ctx]
        try:
            return [str(c) for c in ctx if c is not None]
        except TypeError:
            return [str(ctx)]

    # Helper 2: light tokenizer (used for both sentence and context)
    def _fact_tokens(self, text: str):
        # Lowercase alphanumeric-ish tokens; keeps it dependency-free and robust to punctuation
        return re.findall(r"[a-zA-Z0-9°Ω]+", (text or "").lower())

    # Helper 3: decide if ONE sentence is supported by ANY context
    def _fact_supported(self, sentence: str, contexts: list, overlap_threshold: float, min_phrase_hits: int) -> bool:
        """
        A sentence is supported if:
        - It contains at least `min_phrase_hits` context phrases (for short contexts), OR
        - Its unigram/bigram Jaccard overlap with ANY (long) context >= overlap_threshold.
        """
        s_lower = (sentence or "").lower()
        s_tokens = self._fact_tokens(sentence)

        # Quick exit for empty sentence
        if not s_tokens:
            return False

        # Count phrase hits for short contexts (<= 8 tokens)
        phrase_hits = 0
        for ctx in contexts:
            ctx_lower = (ctx or "").lower().strip()
            if not ctx_lower:
                continue
            ctx_tokens = self._fact_tokens(ctx_lower)

            if len(ctx_tokens) <= 8:  # treat as phrase-like
                # Prefer word-boundary match; fall back to substring for punctuation/markdown cases
                pattern = r'\b' + re.escape(ctx_lower) + r'\b'
                if re.search(pattern, s_lower) or ctx_lower in s_lower:
                    phrase_hits += 1
                    if phrase_hits >= min_phrase_hits:
                        return True

    # If phrase hits not enough, fall back to lexical overlap vs long contexts
        def jaccard(a, b):
            A, B = set(a), set(b)
            if not A and not B:
                return 1.0
            if not A or not B:
                return 0.0
            return len(A & B) / len(A | B)

        def bigrams(tokens):
            return [" ".join(tokens[i:i+2]) for i in range(len(tokens) - 1)] if len(tokens) >= 2 else []

        s_bigrams = bigrams(s_tokens)

        for ctx in contexts:
            ctx_tokens = self._fact_tokens(ctx)
            # Heuristic: treat longer contexts as passages for overlap scoring
            if len(ctx_tokens) > 8:
                c_bigrams = bigrams(ctx_tokens)
                uni = jaccard(s_tokens, ctx_tokens)
                bi = jaccard(s_bigrams, c_bigrams)
                overlap = 0.7 * uni + 0.3 * bi
                if overlap >= overlap_threshold:
                    return True

        return False

    # Main API: compute FActScore (proxy)
    def compute_factscore(
        self,
        generated: str,
        context_text,                      # str or List[str]
        overlap_threshold: float = 0.35,
        min_tokens_per_fact: int = 5,
        min_phrase_hits: int = 1
    ) -> Dict[str, float]:
        """
        Split generation into 'atomic facts' (short sentences) and compute:
            factscore_score = supported_facts / total_facts

        Supported if:
        - sentence matches ≥ min_phrase_hits context phrases (short contexts), OR
        - unigram+bigram Jaccard overlap with any long context ≥ overlap_threshold.
        """
        contexts = self._fact_normalize_context(context_text)
        if not generated or not contexts:
            return {"factscore_supported": 0, "factscore_total_facts": 0, "factscore_score": 0.0}

        # Slice into atomic sentences; ignore very short fragments
        sentences = re.split(r"(?<=[.!?])\s+|\n+", generated.strip())
        atomic = [s for s in sentences if len(self._fact_tokens(s)) >= min_tokens_per_fact]
        if not atomic:
            return {"factscore_supported": 0, "factscore_total_facts": 0, "factscore_score": 0.0}

        supported = sum(
            1 for s in atomic
            if self._fact_supported(s, contexts, overlap_threshold, min_phrase_hits)
        )

        score = supported / len(atomic)
        return {
            "factscore_supported": supported,
            "factscore_total_facts": len(atomic),
            "factscore_score": round(score, 4)
        }


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

    def compute_text_f1(
        self,
        generated: str,
        reference: str,
        case_sensitive: bool = False
    ) -> Dict[str, float]:
        """
        Compute F1 score based on word overlap between generated and reference text.
        Automatically compares all words - no manual keyword list needed.

        Args:
            generated: Model-generated response
            reference: Reference/correct answer
            case_sensitive: Whether to match case-sensitively

        Returns:
            Dictionary with precision, recall, F1, and word overlap details

        Example:
            generated: "The cell divides during mitosis through prophase and metaphase"
            reference: "Mitosis involves prophase, metaphase, and anaphase stages"

            Result: {
                'precision': 0.3333,  # 3 matches / 9 generated words
                'recall': 0.5000,     # 3 matches / 6 reference words
                'f1': 0.4000,
                'matched_words': ['mitosis', 'prophase', 'metaphase'],
                'missing_words': ['anaphase', 'stages', 'involves'],
                'extra_words': ['cell', 'divides', 'during', 'through', 'and']
            }
        """
        # Tokenize and normalize
        def get_words(text: str) -> List[str]:
            words = re.findall(r'\b\w+\b', text)
            return words if case_sensitive else [w.lower() for w in words]

        gen_words = get_words(generated)
        ref_words = get_words(reference)

        if not ref_words:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'matched_words': [],
                'missing_words': [],
                'extra_words': gen_words,
                'generated_word_count': len(gen_words),
                'reference_word_count': 0
            }

        # Convert to multisets (Counter) to handle repeated words correctly
        gen_counter = Counter(gen_words)
        ref_counter = Counter(ref_words)

        # Find intersection (matched words with min count)
        matched_counter = gen_counter & ref_counter
        matched_words = list(matched_counter.elements())

        # Find words in reference but not in generated (missing)
        missing_counter = ref_counter - gen_counter
        missing_words = list(missing_counter.elements())

        # Find words in generated but not in reference (extra/false positives)
        extra_counter = gen_counter - ref_counter
        extra_words = list(extra_counter.elements())

        # Calculate metrics
        true_positives = len(matched_words)
        false_positives = len(extra_words)
        false_negatives = len(missing_words)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'matched_words': matched_words,
            'missing_words': missing_words,
            'extra_words': extra_words,
            'generated_word_count': len(gen_words),
            'reference_word_count': len(ref_words)
        }

    def compute_keyword_recall(
        self,
        generated: str,
        expected_keywords: List[str],
        case_sensitive: bool = False
    ) -> Dict[str, any]:
        """
        Compute keyword coverage using simple recall metric.
        Measures what percentage of expected keywords appear in the response.

        Args:
            generated: Model-generated response
            expected_keywords: List of keywords that should appear
            case_sensitive: Whether to match case-sensitively

        Returns:
            Dictionary with recall score and keyword details

        Example:
            {
                'recall': 0.80,          # 4 out of 5 keywords found
                'matched_keywords': ['chlorine', 'bromine', 'reactive', 'displace'],
                'missing_keywords': ['halogen'],
                'match_count': 4,
                'expected_count': 5
            }
        """
        if not expected_keywords:
            return {
                'recall': 0.0,
                'matched_keywords': [],
                'missing_keywords': [],
                'match_count': 0,
                'expected_count': 0
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

        # Calculate recall
        recall = len(matched) / len(keywords) if keywords else 0.0

        return {
            'recall': round(recall, 4),
            'matched_keywords': matched,
            'missing_keywords': missing,
            'match_count': len(matched),
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
        context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compute all available metrics for a response.
        (Note: FActScore needs context_text; use compute_factscore separately.)
        """
        metrics = {
            'response_length': len(generated),
            'word_count': len(generated.split())
        }

        if reference:
            metrics['rouge'] = self.compute_rouge(generated, reference)
            metrics['text_f1'] = self.compute_text_f1(generated, reference)
            metrics['order'] = self.compute_order_score(generated, reference)
            if is_mcq:
                metrics['exact_match'] = self.compute_exact_match(generated, reference)

        if expected_keywords:
            metrics['keyword_recall'] = self.compute_keyword_recall(generated, expected_keywords)

        if context:
            metrics['fact_score'] = self.compute_factscore(generated, context)

        return metrics


# Convenience function for quick evaluation
def evaluate_response(
    generated: str,
    reference: Optional[str] = None,
    expected_keywords: Optional[List[str]] = None,
    is_mcq: bool = False,
    context: Optional[str] = None
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
        is_mcq=is_mcq,
        context=context
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

    # Test case 2: Text F1
    print("\n=== TEST 2: Text F1 ===")
    text_f1 = metrics.compute_text_f1(generated, reference)
    print(f"Text F1: {text_f1}")

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

    # Test case 5: FActScore Testing
    generated = """ D **Both substances contain particles held together by strong electrostatic forces of attraction.** In metals, there are strong electrostatic forces of attraction between the positive metal ions and the delocalized sea of electrons (metallic bonding). In ionic compounds, there are strong electrostatic forces of attraction between oppositely charged ions in the crystal lattice (ionic bonding). This statement is **correct** for both types of substances.
    """
    context = ["electrostatic forces of attraction", "metallic bonding", "ionic bonding", "delocalized electrons", "positive metal ions", "oppositely charged ions", "crystal lattice"]
    fact_scores = metrics.compute_factscore(generated, context)
    print(f"Generated: {generated}")
    print(f"Reference: {context}")
    print(f"FAct Scores for concise answer: {fact_scores}")

    yap_generated = """
    D Both substances contain particles held together by strong electrostatic forces of attraction.
    Let's evaluate each statement:
    A **Both substances are hard and rigid.** Ionic compounds are typically hard and rigid (brittle). While metals can be hard, they are also typically malleable and ductile, meaning they can be deformed without breaking, which contradicts being strictly 'rigid'. So, this statement is not universally correct for both.
    B **Both substances are pure compounds.** The metallic structure shown represents an element (a pure metal) unless specified as an alloy. Ionic substances are compounds. Therefore, stating both are pure compounds is incorrect.
    C **Both substances can conduct electricity in the solid state.** Metals conduct electricity in the solid state due to the mobility of delocalized electrons. However, ionic compounds do not conduct electricity in the solid state because their ions are fixed in the lattice and cannot move; they only conduct when molten or dissolved in solution. So, this statement is incorrect.
    D **Both substances contain particles held together by strong electrostatic forces of attraction.** In metals, there are strong electrostatic forces of attraction between the positive metal ions and the delocalized sea of electrons (metallic bonding). In ionic compounds, there are strong electrostatic forces of attraction between oppositely charged ions in the crystal lattice (ionic bonding). This statement is **correct** for both types of substances.
    """
    yap_fact_scores = metrics.compute_factscore(yap_generated, context)
    print(f"Generated: {yap_generated}")
    print(f"Reference: {context}")
    print(f"FAct Scores for extensive answer: {yap_fact_scores}")

    # Test case 5: All metrics
    print("\n=== TEST 5: All Metrics ===")
    all_metrics = metrics.compute_all_metrics(
        generated=generated,
        reference=reference
    )
    print(f"All metrics keys: {list(all_metrics.keys())}")
    print(f"Order score included: {all_metrics['order']['order_score']}")
