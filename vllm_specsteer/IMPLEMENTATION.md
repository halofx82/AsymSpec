# AsymSpec vLLM Implementation Notes

This document describes the AsymSpec integration for vLLM 0.28.0, focusing on efficient compressed-context drafter evaluation (Path B) and its mathematical equivalence to full re-prefill.

## 1. Algorithm recap

Per speculation step, AsymSpec requires three forwards:

| Forward | Model | Input | Purpose |
|---|---|---|---|
| **drafter** | SLM_aug | aug_prompt (full ctx) | propose K draft tokens + aug_logits |
| **verifier** | LLM | main_prompt (compressed ctx) | target_logits at K+1 positions |
| **base** | SLM_base (= SLM_aug, same weights) | main_prompt (same as verifier) | base_logits at K+1 positions — used in γ-rule + δ-fusion |

The sampler then runs:

```
accept_i  iff  p_llm(draft_i) > γ · p_base(draft_i)
on reject: emit argmax( log p_llm + β · (log p_aug − log p_base) )
```

The third forward (`SLM_base on main_prompt`) is what distinguishes AsymSpec from standard speculative decoding: it provides the drafter's compressed-context reference for CDA and delta fusion.

## 2. Two implementation strategies for the base forward

### Strategy A — naive `dual_base`

At each speculation step *t*, fully re-prefill SLM_base on the entire `main_prompt[0..L_t-1] + drafts`:

```python
for step t:
    h_main = SLM_base.forward(input_ids = main_prompt[0..L_t+K-1])
    base_logits[t] = compute_logits(h_main[L_t-1 : L_t+K])
```

Cost per step: **O(L_main + K)** — the SLM_base re-prefills the entire main context every step.

### Strategy B — incremental KV cache (`PathB`, our default)

Maintain SLM_base's per-layer KV cache `(K_t, V_t)` across steps. At step *t*, only forward the K newly-drafted tokens:

```python
# At init: SLM_base prefills main_prompt[0..L_0-1] once into (K_0, V_0)
for step t:
    new_tokens = drafts ∪ {LLM_bonus_token}
    h_new, (K_{t+1}, V_{t+1}) = SLM_base.forward(
        input_ids   = new_tokens,
        positions   = [L_t .. L_t + K],
        kv_cache    = (K_t, V_t),
    )
    base_logits[t] = compute_logits(h_new)
```

Cost per step: **O(K)** — independent of L_main.

In our vLLM fork this is invoked as `SpecSteerProposer._base_parallel_verify(drafts, next_token_ids)`. SLM_base lives in its own KV cache group (gid=0, layer-prefix `specsteer_base.*`), sharing block tables with the LLM verifier but with separate cache tensors.

## 3. Mathematical equivalence (A ≡ B)

**Claim**: For identical `input_ids[0..L+K-1]` and weights, Strategies A and B produce identical `logits[L..L+K-1]`, modulo bf16 precision (~1e-3 max abs diff).

### Lemma 1 — KV-cache sufficiency
At any layer, hidden state at position *i*:

```
h_i = TransformerLayer(emb_i, K[0..i], V[0..i])
attention(q_i, K, V) = softmax(q_i K^T / √d) V
```

depends *only on the values of `K[0..i]` and `V[0..i]`*, not on how those tensors were produced (whether by full prefill or incremental update).

### Lemma 2 — KV-cache reconstructibility
For fixed weights and input_ids[0..i-1], the KV tensors `K[0..i-1], V[0..i-1]` are deterministic functions of input_ids:

```
K_layer = W_K · X_layer
V_layer = W_V · X_layer
```

where X_layer is recursively determined by lower layers and ultimately by input_ids and position embeddings.

### Theorem — Strategy A ≡ Strategy B
Given identical `input_ids[0..L+K-1]`:

- By **Lemma 2**, the KV cache that Strategy B has accumulated by step *t* for positions `0..L_t-1` is *bit-identical* to what Strategy A would compute by full prefill on `input_ids[0..L_t-1]` (same weights, same inputs ⇒ same outputs).
- By **Lemma 1**, hidden states at positions `L..L+K-1` depend only on (a) embeddings at those positions and (b) KV at all positions `0..L+K-1`.
- Both strategies have identical KV at all relevant positions ⇒ identical hidden states ⇒ identical `compute_logits` output. ∎

This is the standard incremental-decoding equivalence that every production transformer inference engine relies on. PathB simply applies it to SLM_base (which the original SpecSteer paper described in the naive A form).

## 4. Numerical precision

The equivalence holds exactly in real arithmetic. In bf16:

| Source | Magnitude |
|---|---|
| Accumulated KV bf16 round-off | ~1e-4 per layer × N layers ≈ 1e-3 in logits |
| `softmax((q K^T) / √d)` numerical stability | identical between A and B (same formula) |
| KV cache layout (block-table indirection) | identical layout, identical results |
| **Total max abs diff in logits A vs B** | ~1e-3 (verified empirically on Qwen3-32B + Qwen3-4B) |

This is the same precision regime as the LLM verifier's own incremental KV cache — well below any threshold that would affect γ-rule accept/reject decisions (which compare relative log probabilities differing by O(1)).

## 5. Cost comparison

For a single speculation step:

| | Strategy A | Strategy B (PathB) | Speedup |
|---|---|---|---|
| SLM_base FLOPs | O(L_main + K) | **O(K)** | (L_main + K) / K |
| L_main = 16K, K = 2 | 16002 token-fwd | **2 token-fwd** | **~8000×** |
| L_main = 1K, K = 2 | 1002 token-fwd | **2 token-fwd** | ~500× |
| L_main = 100, K = 4 | 104 token-fwd | **4 token-fwd** | 26× |

The SLM_base forward is ~3-15% of total inference time depending on (L_main, K, drafter size). PathB removes ~99% of *that* cost, translating to ~3-15% total wall-clock speedup over Strategy A.

In agentic / long-context workloads (L_main large), PathB is essential — Strategy A would dominate runtime.

## 6. vLLM integration details

### KV cache groups
SLM_base is registered in vLLM's spec-decode pipeline as a separate model with layer prefix `specsteer_base.*`:

- **drafter** (SLM_aug): `gid=1`, own KV cache (sees aug_prompt)
- **base** (SLM_base): `gid=0` layers `specsteer_base.layers.N.*`, distinct from LLM
- **verifier** (LLM): `gid=0` layers `model.layers.N.*`

Without `specsteer_main_max_model_len`, both groups retain vLLM's legacy shared
BlockPool. With the option set, each group has an independent BlockPool and
group-local block IDs. The worker already indexes a block table by group before
passing it to that group's attention layers, so IDs never cross pool boundaries.
Every model layer has its own physical cache tensor sized from its group's block
count; `reshape_and_cache` writes therefore never overlap.

The compressed 4B module tree is created on the meta device with
`initialize_model()` and never invokes a model loader. Parameters and buffers
are rebound to the normally loaded drafter by every qualified registration
name, including duplicate names needed to preserve tied-weight aliases. The
attention objects remain distinct and are registered under `specsteer_base.*`.

### Per-step flow

```
SpecSteerProposer.propose(args, kwargs):
    # Draft stage: K-token drafter forward (already incremental via vLLM)
    draft_token_ids = super().propose(...)            # standard SpS path

    # Base stage: Path B evaluation (the optimization in this document)
    pv_logits = self._base_parallel_verify(            # K+1 logits at L..L+K
        drafts_flat,
        next_token_ids=next_token_ids,                 # LLM-sampled bonus
    )
    self._base_logits_per_pos = [pv_logits[:, k, :] for k in range(K)]

    # The sampler (specsteer_greedy_sample) reads:
    #   target_logits   = LLM verifier's logits      (incremental)
    #   aug_logits      = drafter's K logits          (incremental)
    #   base_logits     = pv_logits                  ← PathB output (incremental)
    return draft_token_ids
```

`_base_parallel_verify` extends SLM_base's KV cache by forwarding the K drafted tokens at positions `[L..L+K-1]`, plus the bonus token at `L+K`. The cache is committed iff the verifier accepts the corresponding draft (via `accept_mask` post-processing, identical to LLM's KV cache management).

### Correctness checks in code

The implementation logs equivalence diagnostics (sampled, not every step):

```
SpecSteer hidden_diag: rel_idx=L start_pos=0 num_committed=L
    my_top1=58 drafter_top1=26117  hidden[L].mean=2.225 .norm=168.473
SpecSteer PathB step t: B=1 K=2 aug=151645 pv=330  |pv-aug|=13.250
```

`|pv-aug|` shows the **expected** difference between SLM-on-aug and SLM-on-main (different contexts ⇒ different logits — this is the whole point of SpecSteer). When PathB is implementation-correct, this difference is positive and stable across steps.

A numerical regression check can compare Path B output against Strategy A for a small number of steps, asserting `|pv_A - pv_B|_max < 1e-3` when modifying speculative-decoding internals.

## 7. Why this is the default

There is no semantic difference between A and B, only an efficiency difference. Strategy A is included to make the cache optimization explicit; practical deployments should use Path B.

PathB is on by default (`_pathb_skip_dual_base = True`) and we recommend leaving it on. The dual-base path is retained only for regression testing.

## 8. Efficient implementation summary

A 1-2 paragraph treatment in the "Efficient Implementation" section:

AsymSpec maintains the base drafter's KV cache across speculation steps and forwards only the newly drafted tokens, reducing the per-step base pass from O(L_main + K) to O(K). The full-prefill and cached formulations produce mathematically equivalent logits at the drafted positions, modulo ordinary finite-precision effects, by the standard KV-cache sufficiency property of causal self-attention.

## References

- Release implementation: `vllm_specsteer/vllm_0_28/specsteer_model.py`
- vLLM incremental decoding (general): `vllm/v1/worker/gpu_model_runner.py` — same KV cache pattern applied to LLM verifier
- Code: `SpecSteerProposer._base_parallel_verify` (line ~806) and `SpecSteerProposer.propose` (line ~2114)
