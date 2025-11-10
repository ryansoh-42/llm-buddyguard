# src/evaluation.py
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import re
from src.metrics import ResponseMetrics

class ModelEvaluator:
    """
    Evaluates model performance on educational tutoring tasks.
    Integrates with ResponseMetrics for ROUGE, Keyword F1, and other quality metrics.
    """

    def __init__(self):
        self.metrics = {}
        self.response_metrics = ResponseMetrics()
    
    def evaluate_relevance(self, response: str, expected_keywords: List[str]) -> float:
        response_lower = (response or "").lower()
        matches = sum(1 for keyword in (expected_keywords or []) if str(keyword).lower() in response_lower)
        return matches / len(expected_keywords) if expected_keywords else 0.0
    
    def check_no_direct_answer(self, response: str) -> bool:
        direct_answer_patterns = [
            r"the answer is",
            r"the solution is",
            r"x\s*=\s*\d+",
            r"answer:\s*\d+",
            r"result:\s*\d+"
        ]
        response_lower = (response or "").lower()
        for pattern in direct_answer_patterns:
            if re.search(pattern, response_lower):
                return False
        return True
    
    def evaluate_step_by_step(self, response: str) -> float:
        step_indicators = [
            "first", "second", "third", "next", "then",
            "step 1", "step 2", "step 3",
            "let's", "can you", "try to"
        ]
        response_lower = (response or "").lower()
        matches = sum(1 for indicator in step_indicators if indicator in response_lower)
        return min(matches / 3.0, 1.0)
    
    def evaluate_tone(self, response: str) -> float:
        encouraging_phrases = [
            "great question", "good thinking", "you're on the right track",
            "well done", "excellent", "that's correct", "nice work"
        ]
        response_lower = (response or "").lower()
        has_encouragement = any(phrase in response_lower for phrase in encouraging_phrases)
        words = (response or "").split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        complexity_penalty = 1.0 if avg_word_length < 7 else 0.8
        return (1.0 if has_encouragement else 0.7) * complexity_penalty
    
    def evaluate_confidence(self, confidence_metrics: Dict) -> Dict:
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
        
        uncertain_token_ratio = high_uncertain_tokens / total_tokens if total_tokens > 0 else 0
        needs_review = False
        review_reasons = []
        if confidence < 0.6:
            needs_review = True
            review_reasons.append(f"Low confidence ({confidence:.2f})")
        if uncertainty > 0.4:
            needs_review = True
            review_reasons.append(f"High avg uncertainty ({uncertainty:.2f})")
        if uncertain_token_ratio > 0.2:
            needs_review = True
            review_reasons.append(f"Many uncertain tokens ({uncertain_token_ratio:.1%})")
        
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
    
    def _compute_phrase_coverage(self, response: str, context_text: Union[str, List[str]]) -> Dict[str, float]:
        """
        Optional companion to FActScore: percentage of context phrases found anywhere in the response.
        Robust to context being str or list[str].
        """
        if context_text is None:
            return {"phrase_coverage": 0.0, "phrase_hits": 0, "phrase_total": 0}
        ctx_list = [context_text] if isinstance(context_text, str) else [c for c in context_text if c]
        if not ctx_list:
            return {"phrase_coverage": 0.0, "phrase_hits": 0, "phrase_total": 0}
        resp = (response or "").lower()
        hits = 0
        for p in ctx_list:
            p = str(p).lower().strip()
            if not p:
                continue
            pattern = r'\b' + re.escape(p) + r'\b'
            if re.search(pattern, resp) or (p in resp):
                hits += 1
        return {
            "phrase_coverage": round(hits / len(ctx_list), 4),
            "phrase_hits": hits,
            "phrase_total": len(ctx_list)
        }

    def evaluate_response(
        self,
        response: str,
        confidence_metrics: Dict = None,
        expected_keywords: List[str] = None,
        reference_answer: Optional[str] = None,
        is_mcq: bool = False,
        context_text: Optional[Union[str, List[str]]] = None,
        factscore_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single response.
        Includes ROUGE (if reference), keyword F1, exact match (MCQ),
        optional FActScore (if context_text provided), confidence, and tutor heuristics.
        """
        # Base tutor heuristics
        results: Dict[str, any] = {
            "no_direct_answer": 1.0 if self.check_no_direct_answer(response) else 0.0,
            "step_by_step_score": self.evaluate_step_by_step(response),
            "tone_score": self.evaluate_tone(response),
            "response_length": len(response or ""),
            "word_count": len((response or "").split())
        }

        # Legacy relevance score
        if expected_keywords:
            results["relevance_score"] = self.evaluate_relevance(response, expected_keywords)

        # ROUGE (needs reference)
        if reference_answer:
            rouge_scores = self.response_metrics.compute_rouge(response or "", reference_answer or "")
            results["rouge"] = rouge_scores
            results["rouge1_f1"] = rouge_scores["rouge1"]["fmeasure"]
            results["rouge2_f1"] = rouge_scores["rouge2"]["fmeasure"]
            results["rougeL_f1"] = rouge_scores["rougeL"]["fmeasure"]

        # Keyword F1 proxy using keyword recall + text precision against keyword bag
        if expected_keywords:
            try:
                bag = " ".join(expected_keywords)
                text_f1 = self.response_metrics.compute_text_f1(response or "", bag)
                precision_proxy = text_f1["precision"]
            except Exception:
                precision_proxy = 0.0
            recall_result = self.response_metrics.compute_keyword_recall(response or "", expected_keywords)
            recall = recall_result["recall"]
            f1 = (2 * precision_proxy * recall / (precision_proxy + recall)) if (precision_proxy + recall) > 0 else 0.0
            results["keyword_f1"] = {
                "precision": round(precision_proxy, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            }
            results["keyword_f1_score"] = round(f1, 4)
            results["keyword_recall"] = round(recall, 4)
            results["keyword_precision"] = round(precision_proxy, 4)

        # Exact match for MCQ (NEW: add nested mcq.* block)
        if is_mcq and reference_answer:
            exact_match_result = self.response_metrics.compute_exact_match(response or "", reference_answer or "")
            # Keep original flat outputs for backward compatibility
            results["exact_match"] = exact_match_result
            results["mcq_accuracy"] = exact_match_result.get("accuracy", 0.0)

            # New nested namespace for clean CSV flattening
            results["mcq"] = {
                "predicted": exact_match_result.get("predicted") or exact_match_result.get("extracted_answer"),
                "reference": reference_answer,
                "exact_match": {
                    "accuracy": float(exact_match_result.get("accuracy", 0.0))
                }
            }

        # (Optional) Safety score: call only if available
        try:
            safety_result = self.response_metrics.compute_safety_score(response or "")  # if available
            results["safety"] = safety_result
            results["safety_score"] = safety_result.get("safety_score", None)
        except Exception:
            pass

        # Confidence metrics
        if confidence_metrics:
            confidence_eval = self.evaluate_confidence(confidence_metrics)
            results["confidence"] = confidence_eval
            results["confidence_score"] = confidence_eval["confidence_score"]

        # NEW: FActScore (proxy) + coverage if context provided
        if context_text is not None:
            params = factscore_params or {}
            factscore = self.response_metrics.compute_factscore(
                generated=response or "",
                context_text=context_text,
                overlap_threshold=params.get("overlap_threshold", 0.35),
                min_tokens_per_fact=params.get("min_tokens_per_fact", 5),
                min_phrase_hits=params.get("min_phrase_hits", 1)
            )
            results.update(factscore)
            # Companion coverage view
            coverage = self._compute_phrase_coverage(response or "", context_text)
            results.update(coverage)

        # Overall score (unchanged composition)
        base_scores = [
            results["no_direct_answer"],
            results["step_by_step_score"],
            results["tone_score"]
        ]
        if "safety_score" in results and results["safety_score"] is not None:
            base_scores.append(results["safety_score"])
        if "confidence_score" in results:
            base_scores.append(results["confidence_score"])

        results["overall_score"] = sum(base_scores) / len(base_scores)
        return results
    
    def evaluate_dataset(
        self, 
        responses: List[Dict], 
        save_path: str = None
    ) -> pd.DataFrame:
        results = []
        for item in responses:
            evaluation = self.evaluate_response(
                response=item.get('response', ''),
                confidence_metrics=item.get('confidence_metrics'),
                expected_keywords=item.get('expected_keywords'),
                reference_answer=item.get('reference_answer'),
                is_mcq=item.get('is_mcq', False),
                context_text=item.get('context_text'),
                factscore_params=item.get('factscore_params')
            )
            evaluation['prompt'] = item.get('prompt', '')
            evaluation['response'] = item.get('response', '')
            results.append(evaluation)
        
        df = pd.DataFrame(results)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total responses evaluated: {len(df)}")
        if 'overall_score' in df.columns:
            print(f"Average overall score: {df['overall_score'].mean():.2f}")
        if 'no_direct_answer' in df.columns:
            print(f"No direct answer rate: {df['no_direct_answer'].mean():.2%}")
        if 'step_by_step_score' in df.columns:
            print(f"Avg step-by-step score: {df['step_by_step_score'].mean():.2f}")
        if 'tone_score' in df.columns:
            print(f"Avg tone score: {df['tone_score'].mean():.2f}")
        if 'factscore_score' in df.columns:
            print(f"FActScore proxy mean: {df['factscore_score'].mean():.2f}")
        if 'phrase_coverage' in df.columns:
            print(f"Phrase coverage mean: {df['phrase_coverage'].mean():.2f}")

        # NEW: MCQ accuracy summary (prefer nested; fall back to flat)
        mcq_nested_col = 'mcq.exact_match.accuracy' if 'mcq.exact_match.accuracy' in df.columns else None
        if mcq_nested_col:
            print(f"MCQ exact-match accuracy (mean): {df[mcq_nested_col].mean():.3f}")
        elif 'mcq_accuracy' in df.columns:
            print(f"MCQ exact-match accuracy (mean): {df['mcq_accuracy'].mean():.3f}")

        # Confidence section (if present)
        if {'raw_confidence', 'uncertainty', 'avg_entropy', 'needs_review'} <= set(df.columns):
            print(f"\n=== CONFIDENCE METRICS ===")
            print(f"Avg confidence: {df['raw_confidence'].mean():.2f}")
            print(f"Avg uncertainty: {df['uncertainty'].mean():.2f}")
            print(f"Avg entropy: {df['avg_entropy'].mean():.2f}")
            print(f"Responses needing review: {int(df['needs_review'].sum())}/{len(df)}")
        
        return df


if __name__ == "__main__":
    # Minimal smoke test
    evaluator = ModelEvaluator()
    test_response = "Answer: C\nBecause both parents must be heterozygous (Gg)."
    results = evaluator.evaluate_response(
        response=test_response,
        reference_answer="C",
        expected_keywords=["heterozygous", "recessive", "dominant"],
        is_mcq=True
    )
    print(results)
