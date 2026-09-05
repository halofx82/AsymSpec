# SPDX-License-Identifier: Apache-2.0
# AsymSpec sampler — γ-rule + δ-fusion, CMA variants.
#
# Active methods (ASYMSPEC_METHOD env var):
#   gamma_rule  — accept iff p_llm > γ·p_base; on reject emit argmax(t + β·δ)
#   cma         — γ_eff = γ·exp(-λ·KL(p_aug‖p_base));  env CMA_LAMBDA (default 1.0)
#   jsd         — γ_eff = γ·exp(-λ·JSD(p_aug,p_base));  env JSD_LAMBDA (default 1.0)
#                 JSD ∈ [0, ln2≈0.693] — bounded, fixes K=4 over-acceptance vs KL
#   jsd_pos     — γ_eff = γ·exp(-JSD/(pos+1)); no hyperparameter, position-adaptive
#                 directly counters K=4 causal contamination: λ_eff decays with pos
#   cma_vnorm   — γ_eff = γ·exp(-KL/log(V)); normalizes KL by vocab entropy (log V)
#                 λ_eff = 1/log(V) ≈ 0.096 for V=32000; zero free parameters
#   cma_hbase   — γ_eff = γ·exp(-KL/H(p_base)); normalizes by per-position base entropy
#                 position-adaptive: lenient when base is uncertain, strict when confident
#
# δ-source ablation (ASYMSPEC_DELTA_SRC env var, default "ours"):
#   ours     — δ = log_softmax(aug) - log_softmax(base)  [contrastive cross-context; default]
#   raw_aug  — δ = log_softmax(aug)                       [naive augmented prior, drop base term]
#   scd      — δ = log_softmax(target) - log_softmax(base)[SCD-style two-model expert/amateur,
#                                                          one (main) context; also used by the
#                                                          SCD baseline, bench --mode scd]
#   Only the fused reject-emission token changes; the acceptance gate
#   (KL/JSD of aug‖base) is unaffected, isolating the δ-construction effect.
#
from __future__ import annotations

import os
import torch

from vllm.triton_utils import tl, triton

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1


def specsteer_greedy_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    aug_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Greedy γ-rule + δ-fusion with optional CMA/JSD acceptance.

    Mode: env var ASYMSPEC_METHOD ∈ {gamma_rule, cma, jsd, jsd_pos}.

    CMA knobs (env vars):
        CMA_LAMBDA       (default 1.0)  — KL scale for cma method
        JSD_LAMBDA       (default 1.0)  — JSD scale for jsd method
        CMA_KL_LOG       (path)         — per-step divergence distribution log (all CMA/JSD)
        ASYMSPEC_DELTA_SRC (default ours) — δ construction: ours|raw_aug|scd
    """
    method = os.environ.get("ASYMSPEC_METHOD", "gamma_rule")
    assert method in {"gamma_rule", "cma", "jsd", "jsd_pos", "cma_vnorm", "cma_hbase"}, method

    assert draft_token_ids.ndim == 1
    assert target_logits.ndim == aug_logits.ndim == base_logits.ndim == 2
    assert target_logits.shape == aug_logits.shape == base_logits.shape
    batch_size = len(num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape
    device = target_logits.device

    # Work in fp32; kernels are memory-bound.
    t_log = torch.log_softmax(target_logits.float(), dim=-1)
    a_log = torch.log_softmax(aug_logits.float(), dim=-1)
    b_log = torch.log_softmax(base_logits.float(), dim=-1)
    # δ-source ablation: only the fused reject-emission token changes; the
    # acceptance gate below (KL/JSD of aug‖base) is computed independently.
    _delta_src = os.environ.get("ASYMSPEC_DELTA_SRC", "ours")
    if _delta_src == "ours":
        delta = a_log - b_log
    elif _delta_src == "raw_aug":
        delta = a_log
    elif _delta_src == "scd":
        delta = t_log - b_log
    else:
        raise ValueError(
            f"ASYMSPEC_DELTA_SRC={_delta_src!r} not in {{ours, raw_aug, scd}}")
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    fused = t_log + beta * delta
    fused_argmax = fused.argmax(dim=-1)   # [num_tokens] — used as reject token by all methods

    # Per-position scalar probabilities at the drafted token.
    idx = draft_token_ids.to(torch.int64).unsqueeze(-1)
    p_llm  = t_log.exp().gather(-1, idx).squeeze(-1)
    p_base = b_log.exp().gather(-1, idx).squeeze(-1)

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device,
    )

    if method == "gamma_rule":
        specsteer_greedy_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            fused_argmax, p_llm, p_base, bonus_token_ids,
            float(gamma), max_spec_len,
        )

    elif method == "cma":
        # γ_eff = γ · exp(-λ · KL(p_aug ‖ p_base))
        lam = float(os.environ.get("CMA_LAMBDA", "1.0"))
        kl_ab = (a_log.exp() * (a_log - b_log)).sum(dim=-1).clamp_min(0.0)
        gamma_eff = (float(gamma) * torch.exp(-lam * kl_ab)).to(torch.float32)
        _log_divergence(kl_ab, os.environ.get("CMA_KL_LOG", ""))
        cma_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            fused_argmax, p_llm, p_base, gamma_eff, bonus_token_ids,
            max_spec_len,
        )

    elif method == "jsd":
        # γ_eff = γ · exp(-λ · JSD(p_aug, p_base))
        # JSD ∈ [0, ln2≈0.693]; with λ=1.0: γ_eff ∈ [γ/2, γ] — safe for K=4.
        lam = float(os.environ.get("JSD_LAMBDA", "1.0"))
        jsd_ab = _jsd(a_log, b_log)
        gamma_eff = (float(gamma) * torch.exp(-lam * jsd_ab)).to(torch.float32)
        _log_divergence(jsd_ab, os.environ.get("CMA_KL_LOG", ""))
        cma_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            fused_argmax, p_llm, p_base, gamma_eff, bonus_token_ids,
            max_spec_len,
        )

    elif method == "jsd_pos":
        # γ_eff = γ · exp(-JSD / (pos + 1))  — no hyperparameter.
        # λ_eff = 1/(pos+1): pos=0 → λ=1.0, pos=3 → λ=0.25.
        # Directly counters K=4 causal contamination: later positions
        # accumulated aug-steered context inflates JSD; dividing by (pos+1)
        # cancels that growth without any tuning.
        jsd_ab = _jsd(a_log, b_log)
        pos_idx = _build_pos_idx(num_draft_tokens, num_tokens, device)
        lam_eff = 1.0 / (pos_idx + 1.0)
        gamma_eff = (float(gamma) * torch.exp(-lam_eff * jsd_ab)).to(torch.float32)
        _log_divergence(jsd_ab, os.environ.get("CMA_KL_LOG", ""))
        cma_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            fused_argmax, p_llm, p_base, gamma_eff, bonus_token_ids,
            max_spec_len,
        )

    elif method == "cma_vnorm":
        # γ_eff = γ · exp(-KL(p_aug‖p_base) / log(V))
        # Normalizes KL by vocab entropy (max possible KL), giving λ_eff = 1/log(V).
        # For V=32000: λ_eff ≈ 0.096 ≈ 0.1 — zero free parameters.
        import math
        lam_eff = 1.0 / math.log(vocab_size)
        kl_ab = (a_log.exp() * (a_log - b_log)).sum(dim=-1).clamp_min(0.0)
        gamma_eff = (float(gamma) * torch.exp(-lam_eff * kl_ab)).to(torch.float32)
        _log_divergence(kl_ab, os.environ.get("CMA_KL_LOG", ""))
        cma_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            fused_argmax, p_llm, p_base, gamma_eff, bonus_token_ids,
            max_spec_len,
        )

    elif method == "cma_hbase":
        # γ_eff = γ · exp(-KL(p_aug‖p_base) / H(p_base))
        # Normalizes KL by per-position base entropy: measures aug divergence
        # relative to the base model's own uncertainty at each token position.
        # Position-adaptive: lenient when base is uncertain, strict when confident.
        kl_ab = (a_log.exp() * (a_log - b_log)).sum(dim=-1).clamp_min(0.0)
        h_base = (-(b_log.exp() * b_log).sum(dim=-1)).clamp_min(1e-4)
        gamma_eff = (float(gamma) * torch.exp(-kl_ab / h_base)).to(torch.float32)
        _log_divergence(kl_ab, os.environ.get("CMA_KL_LOG", ""))
        cma_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            fused_argmax, p_llm, p_base, gamma_eff, bonus_token_ids,
            max_spec_len,
        )

    _plog = os.environ.get("ASYMSPEC_PROVENANCE_LOG", "")
    if _plog:
        _log_provenance(_plog, method, output_token_ids, draft_token_ids,
                        num_draft_tokens, cu_num_draft_tokens, bonus_token_ids)
    return output_token_ids


# ── helpers ──────────────────────────────────────────────────────────────────

def _jsd(a_log: torch.Tensor, b_log: torch.Tensor) -> torch.Tensor:
    """JSD(p_aug, p_base) per token position. Result ∈ [0, ln2≈0.693]."""
    p_a = a_log.exp()
    p_b = b_log.exp()
    m = 0.5 * (p_a + p_b)
    log_m = m.clamp_min(1e-30).log()
    return (0.5 * ((p_a * (a_log - log_m)).sum(dim=-1)
                   + (p_b * (b_log - log_m)).sum(dim=-1))
            ).clamp_min(0.0)


def _build_pos_idx(num_draft_tokens: list[int], num_tokens: int,
                   device: torch.device) -> torch.Tensor:
    """0-indexed draft position for each flattened token slot."""
    pos = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    start = 0
    for n in num_draft_tokens:
        if n > 0:
            pos[start:start + n] = torch.arange(n, dtype=torch.float32, device=device)
        start += n
    return pos


def _log_divergence(div: torch.Tensor, path: str) -> None:
    """Append one per-step divergence summary record to path (if set)."""
    if not path:
        return
    import json as _json
    cpu = div.float().cpu()
    n = cpu.numel()
    if n == 0:
        return
    vals = sorted(cpu.tolist())
    record = {
        "n":    n,
        "mean": float(cpu.mean()),
        "std":  float(cpu.std()) if n > 1 else 0.0,
        "p50":  float(vals[n // 2]),
        "p90":  float(vals[int(n * 0.9)]),
        "p99":  float(vals[int(n * 0.99)]),
        "max":  float(cpu.max()),
    }
    with open(path, "a") as f:
        f.write(_json.dumps(record) + "\n")


def _log_provenance(path, method, out_ids, draft_ids, num_draft, cu, bonus):
    """#41 token-provenance (READ-ONLY, env ASYMSPEC_PROVENANCE_LOG only).

    Post-kernel reconstruction of the kernel's own decision — does NOT
    touch output_token_ids or any decision; pure aggregate count. Per
    request: accepted-draft tokens (leading run matching the draft),
    +1 fused-reject token iff a reject occurred (kernel emits one fused
    token then stops), +1 bonus token iff all K drafts accepted. Wrapped
    so instrumentation can never break decoding."""
    try:
        import json as _json
        o = out_ids.tolist()
        d = draft_ids.tolist()
        bsz = len(num_draft)
        cul = cu.tolist()
        acc = fus = bon = 0
        for b in range(bsz):
            n = int(num_draft[b])
            st = 0 if b == 0 else int(cul[b - 1])
            a = 0
            while a < n and o[b][a] == d[st + a]:
                a += 1
            acc += a
            if a < n:
                fus += 1            # one fused token, then kernel stops
            else:
                bon += 1            # all accepted → bonus appended
        rec = {"method": method, "bsz": bsz, "n_draft": int(sum(num_draft)),
               "accepted": acc, "fused_reject": fus, "bonus": bon}
        with open(path, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass   # never let instrumentation perturb decoding


# ── Triton kernels ────────────────────────────────────────────────────────────

@triton.jit(do_not_specialize=["max_spec_len"])
def cma_kernel(
    output_token_ids_ptr, cu_num_draft_tokens_ptr, draft_token_ids_ptr,
    fused_argmax_ptr, p_llm_ptr, p_base_ptr, gamma_eff_ptr, bonus_token_ids_ptr,
    max_spec_len,
):
    """Per-position γ_eff loaded from gamma_eff_ptr. Used by cma, jsd, jsd_pos."""
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            d    = tl.load(draft_token_ids_ptr + start_idx + pos)
            p_l  = tl.load(p_llm_ptr          + start_idx + pos)
            p_b  = tl.load(p_base_ptr          + start_idx + pos)
            f_id = tl.load(fused_argmax_ptr    + start_idx + pos)
            g    = tl.load(gamma_eff_ptr       + start_idx + pos)
            if p_l > g * p_b:
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, d)
            else:
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, f_id)
                rejected = True
    if not rejected:
        b_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, b_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def specsteer_greedy_kernel(
    output_token_ids_ptr, cu_num_draft_tokens_ptr, draft_token_ids_ptr,
    fused_argmax_ptr, p_llm_ptr, p_base_ptr, bonus_token_ids_ptr,
    gamma, max_spec_len,
):
    """Original γ-rule loop (gamma_rule)."""
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            d    = tl.load(draft_token_ids_ptr + start_idx + pos)
            p_l  = tl.load(p_llm_ptr          + start_idx + pos)
            p_b  = tl.load(p_base_ptr          + start_idx + pos)
            f_id = tl.load(fused_argmax_ptr    + start_idx + pos)
            if p_l > gamma * p_b:
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, d)
            else:
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, f_id)
                rejected = True
    if not rejected:
        b_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, b_id)
