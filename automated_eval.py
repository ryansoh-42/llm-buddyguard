"""
automated_eval.py
------------------------------------------------------------
For each subject, this script:
1) Loads the subject-specific HF model
2) Loads the CSV
3) Processes rows into prompts
4) Runs the subject HF model on the CSV (BATCHED)
5) Runs the OpenAI/Azure (Frontier) model on the CSV (per-row)
6) Calls the metrics API (POST /evaluate) on each generated answer
7) Prints average metrics (one block for the HF model, one for OpenAI)
Also writes per-row CSVs for both models.

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


# ---------------- helpers ----------------
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
    """POST to /evaluate with explicit (connect, read) timeouts."""
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
    """Composite headline score (average of common metrics)."""
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


# ---------------- batching for HF (only in this file) ----------------
def hf_generate_batched(
    hf_model: FineTunedModel,
    prompts: List[str],
    subject: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    do_sample: bool = True,
) -> List[str]:
    """
    Batched generation using hf_model.tokenizer and hf_model.model directly.
    Uses LEFT PADDING for decoder-only models (HF guidance) to ensure correct generation. 
    """
    # Fallback to per-row if internals are not exposed
    if not hasattr(hf_model, "tokenizer") or not hasattr(hf_model, "model"):
        return [hf_model.generate(prompt=p, subject=subject, temperature=temperature,
                                  max_new_tokens=max_new_tokens, do_sample=do_sample)["response"]
                for p in prompts]

    import torch
    tokenizer = hf_model.tokenizer
    model = hf_model.model

    # Ensure safe batching defaults for decoder-only LMs (left padding)
    tokenizer.padding_side = "left"       # <- crucial for correct batched generation on decoder-only models
    tokenizer.truncation_side = "left"    # keep the freshest tokens if we must truncate
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    system = hf_model._get_system_prompt(subject) if hasattr(hf_model, "_get_system_prompt") else \
             f"You are an educational AI tutor for {subject}."
    full_prompts = [f"{system}\n\nStudent: {p}\n\nTutor:" for p in prompts]

    # Batch tokenize: pad to longest in batch (left-pad), truncate if needed
    enc = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )

    # Move to device
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    # Single generate() call for the whole batch
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Slice out only the new tokens for each sample using attention_mask
    attn = enc.get("attention_mask")
    if attn is not None:
        input_lens = [int(attn[i].sum().item()) for i in range(attn.size(0))]
    else:
        input_lens = [enc["input_ids"].shape[1]] * out.size(0)

    decoded: List[str] = []
    for i in range(out.size(0)):
        gen_tokens = out[i, input_lens[i]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        decoded.append(text)
    return decoded


# ---------------- core ----------------
def evaluate_subject(
    subject_name: str,
    csv_path: str,
    hf_model_name: str,
    metrics_api_url: str,
    device: str,
    openai_model: str,
    max_rows: Optional[int],
    output_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Full pipeline for a single subject:
    - HF model in batched mode (here in this file)
    - OpenAI per-row
    - Metrics per-row
    - Per-model averages + per-row CSVs
    """
    print(f"\n=== [{subject_name}] ===")

    # 1) load models
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

    n_rows = len(df) if (max_rows is None or int(max_rows) <= 0) else min(int(max_rows), len(df))
    proc = df.head(n_rows).copy()

    # 3) prepare inputs
    prompts = [compose_prompt(r.context_text, r.question_text) for _, r in proc.iterrows()]
    references = [str(r.answer_text) for _, r in proc.iterrows()]
    keywords_list = [parse_keywords(r.keywords) for _, r in proc.iterrows()]

    # 4) HF batched inference
    B = max(1, int(batch_size))
    hf_outputs: List[str] = []
    for start in tqdm(range(0, n_rows, B), total=(n_rows + B - 1) // B, desc=f"HF batches - {subject_name}"):
        end = min(start + B, n_rows)
        batch_prompts = prompts[start:end]
        try:
            out = hf_generate_batched(
                hf_model,
                batch_prompts,
                subject=subject_name,
                temperature=0.7,
                max_new_tokens=512,
                do_sample=True,
            )
        except Exception as e:
            out = [f"[HF Batch Generation Error] {e}"] * (end - start)
        hf_outputs.extend(out)

    # 5) OpenAI per-row (kept simple & synchronous)
    openai_outputs: List[str] = []
    for i in tqdm(range(n_rows), desc=f"OpenAI rows - {subject_name}"):
        prompt = prompts[i]
        try:
            r_fr = frontier.generate(prompt=prompt, subject="Science")
            fr_text = r_fr.get("response", "")
        except Exception as e:
            fr_text = f"[Frontier Generation Error] {e}"
        openai_outputs.append(fr_text)

    # 6) Metrics per-row
    hf_flat_rows: List[Dict[str, float]] = []
    fr_flat_rows: List[Dict[str, float]] = []
    hf_row_records: List[Dict[str, Any]] = []
    fr_row_records: List[Dict[str, Any]] = []

    for i in tqdm(range(n_rows), desc=f"Metrics - {subject_name}"):
        reference = references[i]
        kws = keywords_list[i]

        m_hf = call_metrics_api(metrics_api_url, hf_outputs[i], reference, kws, is_mcq=False)
        hf_flat = flatten_numeric(m_hf)
        hf_flat_rows.append(hf_flat)
        hf_row_records.append({
            "row_index": i,
            "subject": subject_name,
            "model_name": hf_model_name,
            "model_input_prompt": prompts[i],
            "generated_answer": hf_outputs[i],
            "reference_answer": reference,
            "expected_keywords": "|".join(kws),
            **hf_flat
        })

        m_fr = call_metrics_api(metrics_api_url, openai_outputs[i], reference, kws, is_mcq=False)
        fr_flat = flatten_numeric(m_fr)
        fr_flat_rows.append(fr_flat)
        fr_row_records.append({
            "row_index": i,
            "subject": subject_name,
            "model_name": openai_model,
            "model_input_prompt": prompts[i],
            "generated_answer": openai_outputs[i],
            "reference_answer": reference,
            "expected_keywords": "|".join(kws),
            **fr_flat
        })

    # 7) Averages + prints
    hf_avg = mean_by_key(hf_flat_rows)
    fr_avg = mean_by_key(fr_flat_rows)
    hf_headline = make_headline(hf_avg)
    fr_headline = make_headline(fr_avg)

    print("\n-- Averages (HF model, batched) --")
    if not math.isnan(hf_headline):
        print(f"headline_composite: {hf_headline:.4f}")
    for k in sorted(hf_avg.keys()):
        print(f"{k}: {hf_avg[k]:.6f}")

    print("\n-- Averages (OpenAI) --")
    if not math.isnan(fr_headline):
        print(f"headline_composite: {fr_headline:.4f}")
    for k in sorted(fr_avg.keys()):
        print(f"{k}: {fr_avg[k]:.6f}")

    # Per-row CSVs
    os.makedirs(output_dir, exist_ok=True)
    subject_slug = _slug(subject_name)
    hf_csv_path = os.path.join(output_dir, f"{subject_slug}_hf_rows.csv")
    fr_csv_path = os.path.join(output_dir, f"{subject_slug}_openai_rows.csv")
    pd.DataFrame(hf_row_records).to_csv(hf_csv_path, index=False)
    pd.DataFrame(fr_row_records).to_csv(fr_csv_path, index=False)
    print(f"\nSaved per-row CSVs:\n  - {hf_csv_path}\n  - {fr_csv_path}")

    # Cleanup ASAP to reduce RAM
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
    parser = argparse.ArgumentParser(description="Sequential subject evaluation with HF batching (edit-only automated_eval.py).")
    parser.add_argument("--data_dir", type=str, default="./", help="Directory containing the CSV datasets")
    parser.add_argument("--metrics_api_url", type=str, default="http://localhost:8000", help="Metrics API base URL")
    parser.add_argument("--device", type=str, default="auto", help="HF device map (auto/cuda/cpu)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o", help="OpenAI/Azure model name")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows per dataset (<=0 = all)")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs", help="Directory to write per-row CSV results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for HF model generation")
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
                batch_size=args.batch_size,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] {subject_name}: {e}")

    print("\n=== ALL METRICS (averages) ===")
    print(json.dumps(summaries, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
