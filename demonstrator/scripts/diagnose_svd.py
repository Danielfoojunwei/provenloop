#!/usr/bin/env python3
"""Diagnose SVD rank-reduction error for LoRA adapters.

Loads all 3 adapter checkpoints, extracts LoRA A/B matrices,
computes the full singular value spectrum of the combined
R @ A matrix (from QR decomposition of B), and reports
variance contributions per singular value plus cumulative
variance retained at every possible rank truncation.

This reproduces the exact decomposition path used by
InferenceEngine._truncate_lora_svd in inference_engine.py.
"""

import io
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ADAPTERS = {
    "banking_expert": os.path.join(
        BASE_DIR, "demonstrator", "adapters", "banking_expert_rl", "adapter_rl_final.pt"
    ),
    "investment_expert": os.path.join(
        BASE_DIR, "demonstrator", "adapters", "investment_expert_rl", "adapter_rl_final.pt"
    ),
    "shared_attention": os.path.join(
        BASE_DIR, "demonstrator", "adapters", "shared_attention_rl", "adapter_rl_final.pt"
    ),
}


def load_lora_pair(ckpt_path: str):
    """Load the first q_proj LoRA A/B pair from a checkpoint.

    Mirrors the logic in InferenceEngine._load_lora_from_checkpoint:
    pick the first q_proj LoRA A/B pair as the representative weights.
    """
    with open(ckpt_path, "rb") as f:
        buf = io.BytesIO(f.read())
    state = torch.load(buf, map_location="cpu", weights_only=False)
    model_state = state.get("model_state_dict", {})

    lora_a_keys = sorted(k for k in model_state if "lora_A" in k)
    lora_b_keys = sorted(k for k in model_state if "lora_B" in k)

    if not lora_a_keys or not lora_b_keys:
        raise RuntimeError(f"No LoRA keys found in {ckpt_path}")

    # Prefer q_proj pair
    target_a = target_b = None
    for ak in lora_a_keys:
        if "q_proj" in ak:
            target_a = ak
            bk = ak.replace("lora_A", "lora_B")
            if bk in model_state:
                target_b = bk
            break

    if target_a is None:
        target_a = lora_a_keys[0]
        target_b = target_a.replace("lora_A", "lora_B")
        if target_b not in model_state:
            target_b = lora_b_keys[0]

    a = model_state[target_a].float().numpy().astype(np.float64)
    b = model_state[target_b].float().numpy().astype(np.float64)

    return a, b, target_a, target_b, lora_a_keys, lora_b_keys


def analyse_adapter(name: str, ckpt_path: str):
    """Full SVD spectrum analysis for one adapter."""
    print("=" * 80)
    print(f"  ADAPTER: {name}")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 80)

    a, b, ak, bk, all_a_keys, all_b_keys = load_lora_pair(ckpt_path)

    r = a.shape[0]  # current rank
    d = a.shape[1]  # d_model

    print(f"\n  Key A : {ak}")
    print(f"  Key B : {bk}")
    print(f"  lora_A shape: {a.shape}  (rank x d_model = {r} x {d})")
    print(f"  lora_B shape: {b.shape}  (d_model x rank = {b.shape[0]} x {b.shape[1]})")
    print(f"  Total LoRA A/B pairs in checkpoint: {len(all_a_keys)}")

    # --- QR + SVD (matches _truncate_lora_svd exactly) ---
    Q_b, R_b = np.linalg.qr(b)       # Q [d, r],  R [r, r]
    C = R_b @ a                        # C [r, d] — small matrix
    U_c, S_c, Vh_c = np.linalg.svd(C, full_matrices=False)

    total_var = np.sum(S_c ** 2)
    total_frob = np.sqrt(total_var)

    print(f"\n  -- Singular-value spectrum of R @ A  (rank = {len(S_c)}) --")
    print(f"  Total Frobenius norm ||R@A||_F = {total_frob:.6f}")
    print(f"  Total variance (sum sigma^2)   = {total_var:.6f}")

    # Table header
    print(f"\n  {'idx':>4s}  {'sigma_i':>12s}  {'sigma_i^2':>14s}  "
          f"{'% of var':>9s}  {'cum var %':>9s}  {'Frob err if truncated here':>28s}")
    print("  " + "-" * 86)

    cum_var = 0.0
    for i, s in enumerate(S_c):
        var_i = s ** 2
        pct_i = (var_i / total_var * 100) if total_var > 0 else 0.0
        cum_var += var_i
        cum_pct = (cum_var / total_var * 100) if total_var > 0 else 100.0
        # If we truncate to rank = i+1, discarded = total_var - cum_var
        discarded = total_var - cum_var
        frob_err_rel = np.sqrt(discarded) / total_frob if total_frob > 0 else 0.0

        marker = ""
        if i + 1 in (30, 31, 32):
            marker = "  <-- *** HIGHLIGHTED ***"

        print(f"  {i+1:4d}  {s:12.6f}  {var_i:14.6f}  {pct_i:8.4f}%  "
              f"{cum_pct:8.4f}%  {frob_err_rel:28.6f}{marker}")

    # --- Cumulative retained variance for each possible target rank ---
    print(f"\n  -- Cumulative variance retained per target rank --")
    print(f"  {'target_rank':>12s}  {'retained %':>10s}  {'discarded %':>11s}  "
          f"{'Frob rel err':>12s}  {'note':>20s}")
    print("  " + "-" * 75)

    cum_var = 0.0
    rec_99 = None
    rec_95 = None
    for k in range(1, len(S_c) + 1):
        cum_var += S_c[k - 1] ** 2
        retained = cum_var / total_var * 100 if total_var > 0 else 100.0
        discarded_pct = 100.0 - retained
        disc_var = total_var - cum_var
        frob_rel = np.sqrt(disc_var) / total_frob if total_frob > 0 else 0.0

        note = ""
        if k == 30:
            note = "<-- target_lora_rank"
        elif k == r:
            note = "<-- original rank"

        if rec_99 is None and retained >= 99.0:
            rec_99 = k
        if rec_95 is None and retained >= 95.0:
            rec_95 = k

        print(f"  {k:12d}  {retained:9.4f}%  {discarded_pct:10.4f}%  "
              f"{frob_rel:12.6f}  {note:>20s}")

    # --- Highlight the critical singular values ---
    print(f"\n  -- Critical singular values (30th, 31st, 32nd) --")
    for idx in [29, 30, 31]:
        if idx < len(S_c):
            s = S_c[idx]
            var_i = s ** 2
            pct = var_i / total_var * 100 if total_var > 0 else 0.0
            print(f"    sigma[{idx+1:2d}] = {s:12.6f}   "
                  f"var contrib = {pct:.4f}%   "
                  f"(this is the {'kept' if idx < 30 else 'discarded'} "
                  f"boundary at rank-30)")

    # Ratio analysis
    if len(S_c) >= 31:
        ratio_30_31 = S_c[29] / S_c[30] if S_c[30] > 0 else float('inf')
        print(f"\n    Ratio sigma[30]/sigma[31] = {ratio_30_31:.4f}")
        print(f"    (values close to 1.0 mean no natural spectral gap at rank 30)")

    if len(S_c) >= 2:
        ratio_1_last = S_c[0] / S_c[-1] if S_c[-1] > 0 else float('inf')
        print(f"    Condition number sigma[1]/sigma[{len(S_c)}] = {ratio_1_last:.4f}")

    # --- Recommendations ---
    print(f"\n  -- Recommendations --")
    print(f"    Rank retaining >= 99% variance: {rec_99 if rec_99 else 'N/A'}")
    print(f"    Rank retaining >= 95% variance: {rec_95 if rec_95 else 'N/A'}")
    current_target = 30
    if rec_99 and rec_99 > current_target:
        print(f"    WARNING: target_lora_rank={current_target} does NOT retain 99% variance!")
        print(f"             Need at least rank {rec_99} for 99%.")
    if rec_99 and rec_99 <= current_target:
        print(f"    OK: target_lora_rank={current_target} retains >= 99% variance.")

    # Return summary for cross-adapter comparison
    disc_30 = total_var - np.sum(S_c[:30] ** 2)
    frob_30 = np.sqrt(disc_30) / total_frob if total_frob > 0 else 0.0
    return {
        "name": name,
        "rank": r,
        "total_frob": total_frob,
        "singular_values": S_c,
        "frob_error_at_30": frob_30,
        "rank_99pct": rec_99,
        "rank_95pct": rec_95,
    }


def main():
    print("\n" + "#" * 80)
    print("#  SVD Rank-Reduction Diagnostic for ProvenLoop LoRA Adapters")
    print("#" * 80)

    summaries = []
    for name, path in ADAPTERS.items():
        if not os.path.exists(path):
            print(f"\n  SKIP {name}: checkpoint not found at {path}")
            continue
        summary = analyse_adapter(name, path)
        summaries.append(summary)
        print()

    # --- Cross-adapter summary ---
    print("=" * 80)
    print("  CROSS-ADAPTER SUMMARY")
    print("=" * 80)
    print(f"\n  {'Adapter':>25s}  {'Rank':>5s}  {'||DW||_F':>10s}  "
          f"{'Frob err@30':>12s}  {'err %':>7s}  {'Rank>=99%':>10s}  {'Rank>=95%':>10s}")
    print("  " + "-" * 88)
    for s in summaries:
        err_pct = s["frob_error_at_30"] * 100
        print(f"  {s['name']:>25s}  {s['rank']:5d}  {s['total_frob']:10.4f}  "
              f"{s['frob_error_at_30']:12.6f}  {err_pct:6.2f}%  "
              f"{str(s['rank_99pct']):>10s}  {str(s['rank_95pct']):>10s}")

    # --- Root-cause analysis ---
    print("\n" + "=" * 80)
    print("  ROOT-CAUSE ANALYSIS")
    print("=" * 80)

    all_flat = all(s["rank"] == 32 for s in summaries)
    high_error = any(s["frob_error_at_30"] > 0.10 for s in summaries)

    if high_error:
        print("""
  The ~20.5% Frobenius reconstruction error when reducing from rank 32 to 30
  indicates that the last 2 singular values carry a DISPROPORTIONATELY large
  fraction of the total variance.  Possible root causes:

    1. FLAT SINGULAR VALUE SPECTRUM: If the RL fine-tuning produced nearly
       uniform singular values (all sigma_i roughly equal), then each of
       the 32 singular values carries ~3.125% of variance.  Dropping 2
       would discard ~6.25% of variance, giving sqrt(0.0625) ~ 25% Frobenius
       error.  With slight non-uniformity this lands around 20%.

    2. NOISE-DOMINATED TAIL: If the last few singular values are actually
       *larger* than some middle ones due to training noise or
       regularization artifacts, the tail contributes more variance than
       expected.

    3. CONDITION NUMBER: A near-uniform spectrum (low condition number)
       means there is no natural rank gap.  The SVD truncation is most
       effective when there is a clear spectral gap (a sharp drop-off)
       after the target rank.

  RECOMMENDATION: Check the singular value table above.  If the spectrum
  is approximately flat, consider:
    - Keeping rank 32 (no truncation) and accepting the HE cost
    - Using a much more aggressive truncation (e.g. rank 16) where the
      Frobenius error is expected and budgeted for
    - Re-training with explicit low-rank regularization to encourage
      a spectral gap at the desired rank
""")
    else:
        print("\n  All adapters show acceptable reconstruction error at rank 30.\n")


if __name__ == "__main__":
    main()
