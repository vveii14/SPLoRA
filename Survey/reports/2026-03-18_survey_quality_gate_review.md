# Survey Quality Gate Review

Date: 2026-03-18

## Gate Being Checked

1. Novelty boundary against tensorized PEFT and sparse PEFT is explicit and citation-backed.
2. The survey identifies at least one concrete open gap connecting efficiency and adapter-level interpretability.

## Evidence Reviewed

Primary survey artifacts:

1. `Survey/reports/2026-03-18_peft_literature_scope_and_map.md`
2. `Survey/reports/2026-03-18_sparse_tensor_pathways_novelty_memo.md`

Core cited prior art includes:

1. LoRA
2. SoRA
3. LoRETTA
4. LoTR
5. LoRTA
6. Scaling Sparse Fine-Tuning to Large Language Models
7. SPP
8. Boosted Sparse and Low-Rank Tensor Regression
9. Low-Rank Adapting Models for Sparse Autoencoders

## Gate Assessment

### Gate 1: Novelty boundary is explicit and citation-backed

Status: Pass

Reasoning:

The survey now makes three points clearly and with direct prior-art support:

1. Tensorized LLM PEFT already exists, so tensorization alone is not a contribution.
2. Sparse and sparse-plus-low-rank PEFT already exist, so sparsity alone is not a contribution.
3. CP-style tensorization specifically is already represented in nearby LLM work, especially LoRTA, so a CP-plus-sparsity story would likely be viewed as incremental.

This is enough to constrain the ideation stage away from weak problem framings.

### Gate 2: A concrete efficiency-plus-interpretability gap is identified

Status: Pass

Reasoning:

The survey identifies a usable open gap:

Current PEFT literature emphasizes parameter count, memory, and downstream performance, while adapter-level interpretability is weakly specified. Sparse tensor components may offer a mechanism for pathway-level decomposition, but current literature does not appear to make that the main scientific object together with explicit diagnostics such as component concentration, ablation, and stability.

This is concrete enough to drive method ideation and experimental planning.

## Residual Risks

The survey is sufficient to move forward, but three risks remain active:

1. The closest prior art, especially LoRTA, creates a real novelty risk if the eventual method is too close to CP decomposition plus sparsity regularization.
2. Interpretability claims may collapse into anecdotal visualization unless the evaluation protocol is defined rigorously in advance.
3. The project may still need one stronger differentiator beyond "pathway interpretation," such as structured sparsity, component stability, or a cleaner causal diagnostic framework.

## Decision

Survey stage is sufficient to enter ideation.

The project should proceed under the following constraint:

Do not ideate around "tensor plus sparsity" in the abstract. Ideate around "interpretable sparse tensor pathways" with explicit differentiation from LoRTA, LoRETTA, SoRA, and sparse PEFT baselines.

## Recommended Next Step

Start ideation with a constrained comparison among a small number of candidate method families:

1. CP-style sparse pathway adapters.
2. Structured sparse tensor adapters aligned with architectural groups.
3. Interpretability-first PEFT with a minimal new adapter plus a stronger evaluation protocol.
