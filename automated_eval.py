"""
automated_eval.py (HF batching + MCQ letter + explanation metrics; baseline-only option)
---------------------------------------------------------------------------------------
When --baseline_only is provided:
  • Use a single baseline HF model (default: meta-llama/Llama-3.2-1B-Instruct)
  • Skip OpenAI entirely (no generation, no metrics, no CSV for OpenAI)

Otherwise (default behavior):
  • Use subject-specific HF model per dataset
  • Also run OpenAI (Frontier) per row

For each subject:
1) Load HF model (batched generation with left-padding)
2) Load CSV
3) Build prompts
4) Generate with HF (batched) and optionally OpenAI (row-wise)
5) Metrics:
   - If MCQ (is_mcq TRUE and mcq_answer present):
       a) MCQ letter evaluation (is_mcq=True, reference = mcq_answer)
          -> we force a column 'mcq.exact_match.accuracy' even if the API doesn't return it
       b) Explanation evaluation (is_mcq=False, reference = answer_text, +keywords)
   - Else: Explanation evaluation only
6) Print separate averages:
   - MCQ metrics (mcq.*) for HF (and OpenAI if enabled)
   - Explanation metrics (exp.*) for HF (and OpenAI if enabled)
7) Write per-row CSVs with both mcq.* and exp.* columns.
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

# ---------- HTTP session (keep-alive speeds up local API calls) ----------
_SESSION = requests.Session()

# ---------- helpers ----------
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def parse_bool_cell(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "y", "yes"}:
        return True
    if s in {"false", "f", "0", "n", "no"}:
        return False
    return None

def parse_keywords(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    s = str(raw).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(k).strip() for k in obj if str(k).strip()]
    except Exception:
        pass
    parts = re.split(r"[;,|]", s)
    return [p.strip() for p in parts if p.strip()]

def compose_prompt(context_text: str, question_text: str, is_mcq: bool=False) -> str:
    context_text = context_text or ""
    question_text = question_text or ""
    base = f"{context_text}\n\nQuestion: {question_text}".strip()
    if is_mcq:
        # gentle format nudge to stabilize letter extraction
        base += "\n\nPlease answer with a single capital letter (A–D) on the first line, then give a brief explanation."
    return base

def call_metrics_api(
    base_url: str,
    generated: str,
    reference: Optional[str],
    expected_keywords: Optional[List[str]],
    is_mcq: bool = False,
    timeout: Tuple[float, float] = (3.0, 15.0),
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/evaluate"
    payload = {
        "generated": generated or "",
        "reference": (reference or None),
        "expected_keywords": expected_keywords or None,
        "is_mcq": bool(is_mcq),
    }
    try:
        resp = _SESSION.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("metrics", {}) if isinstance(data, dict) else {"_raw_api_response": data}
    except Exception as e:
        return {"_api_error": str(e)}

def flatten_numeric(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    def _walk(x: Any, pfx: str = ""):
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(v, f"{pfx}{k}.")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out[pfx[:-1]] = float(x)
    _walk(d, prefix)
    return out

def mean_by_key_prefix(rows: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if not k.startswith(prefix):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}

def make_headline(flat_avg: Dict[str, float], prefix: str = "") -> float:
    candidates = [
        f"{prefix}text_f1.f1",
        f"{prefix}rouge.rougeL.fmeasure",
        f"{prefix}keyword_recall.recall",
        f"{prefix}factscore_score",
    ]
    vals = [flat_avg[k] for k in candidates if k in flat_avg]
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

# ---- MCQ letter extraction (A/B/C/D) ----
_MCQ_LETTER_RE = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)
def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = _MCQ_LETTER_RE.search(text)
    return m.group(1).upper() if m else None

# ---------- batched HF generation in this file ----------
def hf_generate_batched(
    hf_model: FineTunedModel,
    prompts: List[str],
    subject: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    do_sample: bool = True,
) -> List[str]:
    if not hasattr(hf_model, "tokenizer") or not hasattr(hf_model, "model"):
        return [hf_model.generate(prompt=p, subject=subject, temperature=temperature,
                                  max_new_tokens=max_new_tokens, do_sample=do_sample)["response"]
                for p in prompts]

    import torch
    tokenizer = hf_model.tokenizer
    model = hf_model.model

    # Left-padding for decoder-only models (prevents wrong decoding on pads)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    system = hf_model._get_system_prompt(subject) if hasattr(hf_model, "_get_system_prompt") \
             else f"You are an educational AI tutor for {subject}."
    full_prompts = [f"{system}\n\nStudent: {p}\n\nTutor:" for p in prompts]

    enc = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    attn = enc.get("attention_mask")
    input_lens = [int(attn[i].sum().item()) for i in range(attn.size(0))] if attn is not None \
                 else [enc["input_ids"].shape[1]] * out.size(0)

    decoded: List[str] = []
    for i in range(out.size(0)):
        gen_tokens = out[i, input_lens[i]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        decoded.append(text)
    return decoded

# ---------- core ----------
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
    baseline_only: bool,
) -> Dict[str, Any]:
    print(f"\n=== [{subject_name}] ===")

    # 1) load models
    hf_model = FineTunedModel(model_name=hf_model_name, device=device)

    # Only load Frontier/OpenAI if not baseline-only
    frontier = None
    if not baseline_only:
        frontier = FrontierModel(model_name=openai_model)

    # 2) load csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"context_text", "question_text", "answer_text", "keywords", "is_mcq", "mcq_answer"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")

    n_rows = len(df) if (max_rows is None or int(max_rows) <= 0) else min(int(max_rows), len(df))
    proc = df.head(n_rows).copy()

    # Normalize MCQ flags & answers
    proc["__is_mcq_bool"] = proc["is_mcq"].apply(parse_bool_cell)
    proc["__mcq_answer_str"] = proc["mcq_answer"].apply(lambda x: (str(x).strip().upper() if pd.notna(x) else ""))

    # 3) prepare inputs
    prompts = [compose_prompt(r.context_text, r.question_text, is_mcq=parse_bool_cell(r.is_mcq) is True)
               for _, r in proc.iterrows()]
    references_free = [str(r.answer_text) for _, r in proc.iterrows()]
    keywords_list = [parse_keywords(r.keywords) for _, r in proc.iterrows()]

    eff_is_mcq: List[bool] = []
    mcq_refs: List[Optional[str]] = []
    for _, r in proc.iterrows():
        flag = bool(r["__is_mcq_bool"])
        mcq_ans = r["__mcq_answer_str"]
        if flag and mcq_ans in {"A", "B", "C", "D"}:
            eff_is_mcq.append(True)
            mcq_refs.append(mcq_ans)
        else:
            eff_is_mcq.append(False)
            mcq_refs.append(None)

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

    # 5) OpenAI per-row (only if not baseline-only)
    openai_outputs: List[str] = []
    if not baseline_only:
        for i in tqdm(range(n_rows), desc=f"OpenAI rows - {subject_name}"):
            prompt = prompts[i]
            try:
                r_fr = frontier.generate(prompt=prompt, subject="Science")
                fr_text = r_fr.get("response", "")
            except Exception as e:
                fr_text = f"[Frontier Generation Error] {e}"
            openai_outputs.append(fr_text)

    # 6) Metrics per-row (MCQ + Explanation)
    hf_flat_rows: List[Dict[str, float]] = []
    fr_flat_rows: List[Dict[str, float]] = []
    hf_row_records: List[Dict[str, Any]] = []
    fr_row_records: List[Dict[str, Any]] = []

    def ensure_mcq_accuracy(flat: Dict[str, float], pred: Optional[str], ref: Optional[str]) -> None:
        """
        Guarantee 'mcq.exact_match.accuracy' exists in the flattened dict.
        If API didn't provide nested path, map mcq.accuracy -> mcq.exact_match.accuracy.
        If still missing, compute locally: 1.0 if pred == ref else 0.0 (when both present).
        """
        if "mcq.exact_match.accuracy" in flat:
            return
        if "mcq.accuracy" in flat:
            flat["mcq.exact_match.accuracy"] = float(flat["mcq.accuracy"])
            return
        if pred and ref and pred in {"A", "B", "C", "D"} and ref in {"A", "B", "C", "D"}:
            flat["mcq.exact_match.accuracy"] = 1.0 if pred == ref else 0.0

    for local_idx in tqdm(range(n_rows), desc=f"Metrics - {subject_name}"):
        ref_free = references_free[local_idx]
        kws = keywords_list[local_idx]
        is_mcq = eff_is_mcq[local_idx]
        mcq_ref = mcq_refs[local_idx]

        # --- HF ---
        hf_text = hf_outputs[local_idx]
        mcq_pred = extract_choice_letter(hf_text) if is_mcq else None
        mcq_metrics_hf = {}
        if is_mcq and mcq_ref:
            m = call_metrics_api(args.metrics_api_url, mcq_pred or "", mcq_ref, None, is_mcq=True)
            mcq_metrics_hf = flatten_numeric(m, prefix="mcq.")
        exp_metrics_hf = call_metrics_api(args.metrics_api_url, hf_text, ref_free, kws, is_mcq=False)
        exp_metrics_hf = flatten_numeric(exp_metrics_hf, prefix="exp.")

        flat_hf = {}
        flat_hf.update(mcq_metrics_hf)
        flat_hf.update(exp_metrics_hf)
        ensure_mcq_accuracy(flat_hf, mcq_pred, mcq_ref)

        hf_flat_rows.append(flat_hf)
        hf_row_records.append({
            "row_index": int(proc.index[local_idx]),
            "subject": subject_name,
            "model_name": hf_model_name,
            "is_mcq_effective": is_mcq,
            "mcq.ref": mcq_ref or "",
            "mcq.pred": mcq_pred or "",
            "model_input_prompt": prompts[local_idx],
            "generated_answer": hf_text,
            "reference_explanation": ref_free,
            "expected_keywords": "|".join(kws),
            **flat_hf
        })

        # --- OpenAI (only when enabled) ---
        if not baseline_only:
            fr_text = openai_outputs[local_idx]
            mcq_pred_fr = extract_choice_letter(fr_text) if is_mcq else None
            mcq_metrics_fr = {}
            if is_mcq and mcq_ref:
                m = call_metrics_api(args.metrics_api_url, mcq_pred_fr or "", mcq_ref, None, is_mcq=True)
                mcq_metrics_fr = flatten_numeric(m, prefix="mcq.")
            exp_metrics_fr = call_metrics_api(args.metrics_api_url, fr_text, ref_free, kws, is_mcq=False)
            exp_metrics_fr = flatten_numeric(exp_metrics_fr, prefix="exp.")

            flat_fr = {}
            flat_fr.update(mcq_metrics_fr)
            flat_fr.update(exp_metrics_fr)
            ensure_mcq_accuracy(flat_fr, mcq_pred_fr, mcq_ref)

            fr_flat_rows.append(flat_fr)
            fr_row_records.append({
                "row_index": int(proc.index[local_idx]),
                "subject": subject_name,
                "model_name": openai_model,
                "is_mcq_effective": is_mcq,
                "mcq.ref": mcq_ref or "",
                "mcq.pred": mcq_pred_fr or "",
                "model_input_prompt": prompts[local_idx],
                "generated_answer": fr_text,
                "reference_explanation": ref_free,
                "expected_keywords": "|".join(kws),
                **flat_fr
            })

    # 7) Averages + prints
    hf_mcq_avg = mean_by_key_prefix(hf_flat_rows, "mcq.")
    hf_exp_avg = mean_by_key_prefix(hf_flat_rows, "exp.")
    hf_headline = make_headline(hf_exp_avg, prefix="exp.")

    print("\n-- Averages (HF model) --")
    if not math.isnan(hf_headline):
        print(f"headline_composite (explanation): {hf_headline:.4f}")
    if hf_mcq_avg:
        print("MCQ metrics:")
        for k in sorted(hf_mcq_avg.keys()):
            print(f"{k}: {hf_mcq_avg[k]:.6f}")
    if hf_exp_avg:
        print("Explanation metrics:")
        for k in sorted(hf_exp_avg.keys()):
            print(f"{k}: {hf_exp_avg[k]:.6f}")

    fr_mcq_avg, fr_exp_avg, fr_headline = {}, {}, float("nan")

    if not baseline_only:
        fr_mcq_avg = mean_by_key_prefix(fr_flat_rows, "mcq.")
        fr_exp_avg = mean_by_key_prefix(fr_flat_rows, "exp.")
        fr_headline = make_headline(fr_exp_avg, prefix="exp.")

        print("\n-- Averages (OpenAI) --")
        if not math.isnan(fr_headline):
            print(f"headline_composite (explanation): {fr_headline:.4f}")
        if fr_mcq_avg:
            print("MCQ metrics:")
            for k in sorted(fr_mcq_avg.keys()):
                print(f"{k}: {fr_mcq_avg[k]:.6f}")
        if fr_exp_avg:
            print("Explanation metrics:")
            for k in sorted(fr_exp_avg.keys()):
                print(f"{k}: {fr_exp_avg[k]:.6f}")

    # Per-row CSVs
    os.makedirs(output_dir, exist_ok=True)
    subject_slug = _slug(subject_name)
    hf_csv_path = os.path.join(output_dir, f"{subject_slug}_hf_rows.csv")
    pd.DataFrame(hf_row_records).to_csv(hf_csv_path, index=False)
    print(f"\nSaved per-row CSV:\n  - {hf_csv_path}")

    fr_csv_path = None
    if not baseline_only:
        fr_csv_path = os.path.join(output_dir, f"{subject_slug}_openai_rows.csv")
        pd.DataFrame(fr_row_records).to_csv(fr_csv_path, index=False)
        print(f"  - {fr_csv_path}")

    # Cleanup ASAP to reduce RAM
    free_model(hf_model)
    if frontier is not None:
        free_model(frontier)

    summary = {
        "subject": subject_name,
        "hf_model": hf_model_name,
        "hf_mcq_avg": hf_mcq_avg,
        "hf_exp_avg": hf_exp_avg,
        "hf_headline": hf_headline,
        "hf_rows_csv": hf_csv_path,
    }
    if not baseline_only:
        summary.update({
            "openai_model": openai_model,
            "openai_mcq_avg": fr_mcq_avg,
            "openai_exp_avg": fr_exp_avg,
            "openai_headline": fr_headline,
            "openai_rows_csv": fr_csv_path,
        })
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Sequential subject evaluation with HF batching + MCQ metrics (baseline-only optional)."
    )
    parser.add_argument("--data_dir", type=str, default="./", help="Directory containing the CSV datasets")
    parser.add_argument("--metrics_api_url", type=str, default="http://localhost:8000", help="Metrics API base URL")
    parser.add_argument("--device", type=str, default="auto", help="HF device map (auto/cuda/cpu)")
    parser.add_argument("--openai_model", type=str, default="gpt-4o", help="OpenAI/Azure model name")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows per dataset (<=0 = all)")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs", help="Directory to write per-row CSV results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for HF model generation")

    # NEW: baseline-only switch and model override
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="If set, run ONLY the baseline HF model and skip OpenAI."
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model to use when --baseline_only is set."
    )

    global args  # used in evaluate_subject for metrics URL
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Choose HF model name function per subject based on baseline_only
    def subject_hf_model(subject_default: str) -> str:
        return args.baseline_model if args.baseline_only else subject_default

    subjects = [
        ("Physics",   os.path.join(args.data_dir, "test_dataset_physics_200.csv"), subject_hf_model("Fawl/is469_project_physics")),
        ("Chemistry", os.path.join(args.data_dir, "test_dataset_chem_200.csv"),    subject_hf_model("Fawl/is469_project_chem")),
        ("Biology",   os.path.join(args.data_dir, "test_dataset_bio_200.csv"),     subject_hf_model("Fawl/is469_project_bio")),
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
                baseline_only=args.baseline_only,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] {subject_name}: {e}")

    print("\n=== ALL METRICS (averages) ===")
    print(json.dumps(summaries, indent=2, sort_keys=False))

if __name__ == "__main__":
    main()
