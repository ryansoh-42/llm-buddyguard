"""
automated_eval.py
------------------------------------------
For each subject, this script:
1) Loads the subject-specific HF model
2) Loads the CSV
3) Processes rows into prompts
4) Runs the subject HF model on the CSV
5) Runs the OpenAI/Azure (Frontier) model on the CSV
6) Calls the metrics API (POST /evaluate) on each generated answer
7) Prints average metrics (one block for the HF model, one for OpenAI)
8) Prints a full JSON dump of all averaged metrics.
9) Profit $$$
"""

import os
import re
import gc
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm
import requests

from src.models.finetune import FineTunedModel
from src.models.frontier import FrontierModel


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def parse_keywords(raw: Any) -> List[str]:
    """Parse keywords robustly from JSON-like strings or delimited text."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # Try JSON list first
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(k).strip() for k in obj if str(k).strip()]
    except Exception:
        pass
    # Fallback delimiters
    parts = re.split(r"[;,|]", s)
    return [p.strip() for p in parts if p.strip()]

def compose_prompt(context_text: str, question_text: str) -> str:
    context_text = context_text or ""
    question_text = question_text or ""
    return f"{context_text}\n\nQuestion: {question_text}".strip()

def call_metrics_api(
    base_url: str,
    generated: str,
    reference: Optional[str],
    expected_keywords: Optional[List[str]],
    is_mcq: bool = False,
    timeout: Tuple[float, float] = (3.0, 12.0),
) -> Dict[str, Any]:
    """
    POST to /evaluate with explicit (connect, read) timeouts.
    """
    url = f"{base_url.rstrip('/')}/evaluate"
    payload = {
        "generated": generated or "",
        "reference": (reference or None),
        "expected_keywords": expected_keywords or None,
        "is_mcq": bool(is_mcq),
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("metrics", {}) if isinstance(data, dict) else {"_raw_api_response": data}
    except Exception as e:
        return {"_api_error": str(e)}

def flatten_numeric(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested dicts to dotted keys; keep only numeric values."""
    out: Dict[str, float] = {}
    def _walk(x: Any, pfx: str = ""):
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(v, f"{pfx}{k}.")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out[pfx[:-1]] = float(x)
    _walk(d, prefix)
    return out

def mean_by_key(rows: List[Dict[str, float]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}

def make_headline(flat_avg: Dict[str, float]) -> float:
    """
    Composite headline score (average of available common metrics).
    """
    keys = ["text_f1.f1", "rouge.rougeL.fmeasure", "keyword_recall.recall", "factscore_score"]
    vals = [flat_avg[k] for k in keys if k in flat_avg]
    return float(sum(vals) / len(vals)) if vals else float("nan")

def free_model(obj):
    try:
        import torch
        del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def evaluate_subject(
    subject_name: str,
    csv_path: str,
    hf_model_name: str,
    metrics_api_url: str,
    device: str,
    openai_model: str,
    max_rows: Optional[int],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Runs the full pipeline for a single subject and prints averages.
    Returns a dict of averaged metrics for HF and OpenAI and writes per-row CSVs.
    """
    print(f"\n=== [{subject_name}] ===")

    # 1) load model
    hf_model = FineTunedModel(model_name=hf_model_name, device=device)
    frontier = FrontierModel(model_name=openai_model)

    # 2) load csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"context_text", "question_text", "answer_text", "keywords"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")

    # exact row limit + tqdm total
    n_rows = len(df) if (max_rows is None or int(max_rows) <= 0) else min(int(max_rows), len(df))
    proc = df.head(n_rows).copy()

    # 3-6) process, run models, run metrics
    hf_flat_rows: List[Dict[str, float]] = []
    fr_flat_rows: List[Dict[str, float]] = []

    # detailed per-row outputs for CSV export
    hf_row_records: List[Dict[str, Any]] = []
    fr_row_records: List[Dict[str, Any]] = []

    for i, row in tqdm(proc.iterrows(), total=n_rows, desc=f"Rows - {subject_name}"):
        prompt = compose_prompt(row.get("context_text", ""), row.get("question_text", ""))
        reference = row.get("answer_text", "")
        keywords = parse_keywords(row.get("keywords", ""))

        # 4) specified HF model
        try:
            r_hf = hf_model.generate(prompt=prompt, subject=subject_name)
            hf_text = r_hf.get("response", "")
        except Exception as e:
            hf_text = f"[HF Generation Error] {e}"

        # 5) OpenAI API (Frontier)
        try:
            r_fr = frontier.generate(prompt=prompt, subject="Science")
            fr_text = r_fr.get("response", "")
        except Exception as e:
            fr_text = f"[Frontier Generation Error] {e}"

        # 6) metrics
        m_hf = call_metrics_api(metrics_api_url, hf_text, reference, keywords, is_mcq=False)
        m_fr = call_metrics_api(metrics_api_url, fr_text, reference, keywords, is_mcq=False)

        # Flatten for averaging
        hf_flat = flatten_numeric(m_hf)
        fr_flat = flatten_numeric(m_fr)
        hf_flat_rows.append(hf_flat)
        fr_flat_rows.append(fr_flat)

        # Keep rich per-row outputs for CSV export
        base = {
            "row_index": int(i),
            "subject": subject_name,
            "model_input_prompt": prompt,
            "reference_answer": reference,
            "expected_keywords": "|".join(keywords),
        }
        hf_row_records.append({**base, "model_name": hf_model_name, "generated_answer": hf_text, **hf_flat})
        fr_row_records.append({**base, "model_name": openai_model, "generated_answer": fr_text, **fr_flat})

    # 7) print averages (HF & OpenAI)
    hf_avg = mean_by_key(hf_flat_rows)
    fr_avg = mean_by_key(fr_flat_rows)
    hf_headline = make_headline(hf_avg)
    fr_headline = make_headline(fr_avg)

    print("\n-- Averages (HF model) --")
    if not math.isnan(hf_headline):
        print(f"headline_composite: {hf_headline:.4f}")
    for k in sorted(hf_avg.keys()):
        print(f"{k}: {hf_avg[k]:.6f}")

    print("\n-- Averages (OpenAI) --")
    if not math.isnan(fr_headline):
        print(f"headline_composite: {fr_headline:.4f}")
    for k in sorted(fr_avg.keys()):
        print(f"{k}: {fr_avg[k]:.6f}")

    # Write per-row CSVs
    os.makedirs(output_dir, exist_ok=True)
    subject_slug = _slug(subject_name)
    hf_csv_path = os.path.join(output_dir, f"{subject_slug}_hf_rows.csv")
    fr_csv_path = os.path.join(output_dir, f"{subject_slug}_openai_rows.csv")
    pd.DataFrame(hf_row_records).to_csv(hf_csv_path, index=False)
    pd.DataFrame(fr_row_records).to_csv(fr_csv_path, index=False)
    print(f"\nSaved per-row CSVs:\n  - {hf_csv_path}\n  - {fr_csv_path}")

    # Cleanup models asap to reduce RAM
    free_model(hf_model)
    free_model(frontier)

    return {
        "subject": subject_name,
        "hf_model": hf_model_name,
        "openai_model": openai_model,
        "hf_avg": hf_avg,
        "openai_avg": fr_avg,
        "hf_headline": hf_headline,
        "openai_headline": fr_headline,
        "hf_rows_csv": hf_csv_path,
        "openai_rows_csv": fr_csv_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Sequential subject evaluation without globals.")
    parser.add_argument("--data_dir", type=str, default="./", help="Directory containing the CSV datasets")
    parser.add_argument("--metrics_api_url", type=str, default="http://localhost:8000", help="Metrics API base URL")
    parser.add_argument("--device", type=str, default="auto", help="HF device map (auto/cuda/cpu)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o", help="OpenAI/Azure model name")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows per dataset (<=0 = all)")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs", help="Directory to write per-row CSV results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subjects = [
        ("Physics",   os.path.join(args.data_dir, "test_dataset_physics_200.csv"), "Fawl/is469_project_physics"),
        ("Chemistry", os.path.join(args.data_dir, "test_dataset_chem_200.csv"),    "Fawl/is469_project_chem"),
        ("Biology",   os.path.join(args.data_dir, "test_dataset_bio_200.csv"),     "Fawl/is469_project_bio"),
    ]

    summaries: List[Dict[str, Any]] = []
    for subject_name, csv_path, hf_model_name in tqdm(subjects, desc="Subjects", leave=True):
        try:
            summary = evaluate_subject(
                subject_name=subject_name,
                csv_path=csv_path,
                hf_model_name=hf_model_name,
                metrics_api_url=args.metrics_api_url,
                device=args.device,
                openai_model=args.openai_model,
                max_rows=args.max_rows,
                output_dir=args.output_dir,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] {subject_name}: {e}")

    # Final step: print all metrics as a compact JSON
    print("\n=== ALL METRICS (averages) ===")
    print(json.dumps(summaries, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
