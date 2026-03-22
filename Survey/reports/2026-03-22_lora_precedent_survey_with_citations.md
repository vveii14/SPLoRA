# LoRA Literature Survey for Structured Update Modeling

Date: 2026-03-22

## Scope

This survey checks whether the three ingredients discussed in the current project framing already have direct precedent in the LoRA and PEFT literature:

1. rank-1 or componentized decomposition,
2. regularization over LoRA-style updates,
3. tensor decomposition for LoRA-style adaptation.

The goal is not to prove that the project is novel in the broad sense. The goal is narrower: determine which parts are already established, and identify the smallest defensible claim that remains open.

## Search Procedure and Skill Coverage

This pass was executed under the requested skill mix.

- `academic-researcher`: used as the framing for paper-level comparison and research-gap analysis.
- `inno-deep-research`: used as the synthesis structure for multi-source findings with citations.
- `gemini-deep-research`: attempted, but blocked because `GEMINI_API_KEY` is not set in the environment.
- `biorxiv-database`: executed, but returned no LoRA-relevant results; the returned matches were unrelated biology papers, which is expected because the topic is not bioRxiv-native.
- `dataset-discovery`: executed for benchmark discovery, but the local environment lacked `gh` and Semantic Scholar returned rate limits, so dataset retrieval was incomplete.
- `inno-code-survey`: used to locate and acquire corresponding public codebases for LoTR, LoRTA, LoRETTA, and SPP, which were cloned into `Experiment/code_references/`.

## Bottom-Line Judgment

The literature check supports a fairly sharp conclusion:

- rank-1 decomposition in LoRA-style adaptation: already explored [1][2]
- regularization over LoRA-style adaptation: already explored [3][4]
- tensorized LoRA: already explored [5][6][7]

So the project should **not** be pitched as:

> LoRA + rank-1 decomposition + regularization + tensor decomposition

because each ingredient already has direct precedent.

The remaining defensible claim is narrower:

> make adapter components explicit as structured pathways, regularize them for identifiability at the component level, and treat them as first-class interpretable objects rather than only as a compressed parameterization.

That claim is more consistent with the actual novelty boundary exposed by the current survey.

## 1. Rank-1 or Componentized LoRA: Already Present

The basic algebraic observation is trivial but important:

\[
\Delta W = BA = \sum_{k=1}^{r} b_k a_k^\top.
\]

Standard LoRA already admits a rank-1 expansion [1]. What later work adds is not the identity itself, but a decision to make component structure explicit and useful. AROMA is a direct precedent here: it builds low-rank adaptation through adaptive rank reduction and makes rank structure part of the method rather than a hidden algebraic fact [2]. This means the project cannot claim novelty simply by rewriting LoRA as a sum of rank-1 terms.

What remains open is the **scientific role** assigned to those components. Existing rank-aware work shows that rank-1 or progressively reduced components can be useful. It does not yet force the claim that adapter components should be treated as explicit task pathways whose identity and stability are part of the method objective.

## 2. LoRA Regularization: Already a Real Sub-literature

Regularized LoRA is not a gap. SoRA explicitly couples LoRA with sparsity and adaptive effective rank reduction [3]. SPP also sits firmly in this space by preserving sparsity structure during PEFT [4]. These papers are enough to rule out any claim of novelty based only on ``we add regularization to LoRA.''

The more interesting distinction is the type of regularization. Most existing methods impose sparsity, gating, or parameter-level constraints on the LoRA parameterization. The current project is trying to move toward something more structural:

\[
\Delta W = \sum_{k=1}^{K} \sigma_k \hat{u}_k \hat{v}_k^\top,
\qquad
\mathcal{R}_{\text{mult}} = \sum_{k=1}^{K} |\sigma_k| \|\hat{u}_k\| \|\hat{v}_k\|.
\]

That is closer in spirit to the structure-induced regularization used in boosted sparse and low-rank tensor regression [8] than to standard additive penalties over factor blocks. This distinction matters. If the update is modeled as a set of pathways, then the regularizer should ideally operate on pathways as whole outer-product objects, not only on factors independently.

This does **not** prove that multiplicative pathway regularization is new in LoRA. It only shows that it is not the dominant pattern in the current LoRA regularization literature surveyed here, which leaves room for a cleaner structural claim.

## 3. Tensorized LoRA: Also Already Established

Tensorized PEFT is clearly no longer empty space.

- LoRETTA uses tensor-train structure for ultra-low-parameter fine-tuning [5].
- LoTR uses low tensor-rank weight adaptation and emphasizes cross-layer tensor structure [6].
- LoRTA makes the CP-style tensor adaptation story particularly direct for LLMs [7].

As a result, ``we move from matrix LoRA to tensor LoRA'' is not a publishable novelty statement on its own. At best, tensorization can be justified as the **general setting** in which a stronger idea is instantiated.

The strongest use of tensorization in the current project is therefore not:

> we use tensors because tensors are new

but rather:

> once adapter components are treated as explicit pathways, tensorization is the natural way to align those pathways with architectural modes such as layer groups, heads, or other multiway structure.

That is a much narrower and safer claim.

## 4. Where the Actual Opening Seems to Be

The project still appears viable, but the novelty must sit in the **organization of the ingredients**, not in the ingredients themselves.

The most promising formulation is:

1. start from explicit rank-1 or tensorized components as the primitive update units,
2. regularize these components for identifiability and concentration rather than only for generic shrinkage,
3. evaluate them as first-class interpretable objects through ablation, sparsity, concentration, and stability diagnostics.

This is the point where the project becomes visibly different from a plain ingredient-composition story.

## Comparison Table: Literature Status

| Component | Representative papers | Status | What this means |
| --- | --- | --- | --- |
| Low-rank PEFT baseline | LoRA [1] | Established | Baseline only |
| Rank-1 / componentized LoRA | AROMA [2] | Already explored | Rank-1 decomposition alone is not novel |
| Sparse or regularized LoRA | SoRA [3], SPP [4] | Already explored | Generic LoRA regularization is not novel |
| Tensorized LoRA | LoRETTA [5], LoTR [6], LoRTA [7] | Already explored | Tensorized PEFT is not novel by itself |
| Structure-induced multiplicative regularization intuition | Boosted Sparse and Low-Rank Tensor Regression [8] | Relevant bridge, not direct LoRA method | Useful conceptual support for pathway-level regularization |

## Comparison Table: Defensible Claim Strength

| Claim | Strength | Reason |
| --- | --- | --- |
| ``We decompose LoRA into rank-1 components.'' | Weak | Already implicit in LoRA and explicit in later work [1][2] |
| ``We regularize LoRA.'' | Weak | Already done in several forms [3][4] |
| ``We tensorize LoRA.'' | Weak | Already done [5][6][7] |
| ``We combine rank-1, regularization, and tensorization.'' | Moderate to weak | Ingredients all exist; composition may look incremental |
| ``We treat adapter components as explicit task pathways, regularize them for identifiability, and evaluate them as first-class interpretable objects.'' | Strongest remaining position | Shifts novelty from ingredients to scientific object and evaluation protocol |

## Public Code Repositories Located

Using the code-survey workflow, the following public repos were acquired and checked locally:

| Paper | Repo | Local path | Notes |
| --- | --- | --- | --- |
| LoTR [6] | `daskol/lotr` | `Experiment/code_references/lotr` | README explicitly states low tensor-rank weight adaptation and tensor-structured updates |
| LoRTA [7] | `ihounie/lorta` | `Experiment/code_references/lorta` | Repo includes PEFT integration and task-specific experiment folders |
| LoRETTA [5] | `yifanycc/loretta` | `Experiment/code_references/loretta` | README explicitly positions the method as tensor-train PEFT |
| SPP [4] | `Lucky-Lance/SPP` | `Experiment/code_references/spp` | Useful implementation reference for sparsity-preserved PEFT |

Representative implementation entry points found locally:

- `Experiment/code_references/lotr/lotr/lotr.py`
- `Experiment/code_references/lotr/lotr/lora.py`
- `Experiment/code_references/lorta/peft/src/peft/`
- `Experiment/code_references/loretta/loretta/loretta/utils/tensor_util.py`
- `Experiment/code_references/spp/peft/src/peft/tuners/lora.py`

## Benchmark Ladder Notes

`dataset-discovery` did not return usable results in this environment, so benchmark selection remains a manual step. A reasonable evaluation ladder for this project is still straightforward:

- GLUE for small matrix-setting validation [9]
- SuperGLUE for stronger NLU stress tests [10]
- GSM8K for structured reasoning transfer [11]
- MMLU for broad knowledge and instruction-tuned evaluation [12]

These are not proposed as novel datasets. They are proposed as a clean staged benchmark ladder if the project moves toward experiment design.

## Final Recommendation for Task 9

The first implementation direction should be chosen as follows:

1. start in the matrix setting,
2. expose the update as explicit rank-1 pathways,
3. begin with component sparsity over pathway amplitudes,
4. test a pathway-level multiplicative regularizer,
5. postpone tensorization until the matrix-core claims are isolated.

That is the most defensible route given the current literature boundary.

## Sources

[1] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv 2021. https://arxiv.org/abs/2106.09685  
[2] Lu et al. *AROMA: Adaptive Rank Reduction of Low-Rank Adaptation for Large Language Models*. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.170/  
[3] Ding et al. *Sparse Low-rank Adaptation of Pre-trained Language Models*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.252/  
[4] Lu et al. *SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models*. arXiv 2024. https://arxiv.org/abs/2405.16057  
[5] Yang et al. *LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models*. NAACL 2024. https://aclanthology.org/2024.naacl-long.174/  
[6] Bershatsky et al. *LoTR: Low Tensor Rank Weight Adaptation*. arXiv 2024. https://arxiv.org/abs/2402.01376  
[7] Hounie et al. *LoRTA: Low Rank Tensor Adaptation of Large Language Models*. arXiv 2024. https://arxiv.org/abs/2410.04060  
[8] He et al. *Boosted Sparse and Low-Rank Tensor Regression*. NeurIPS 2018. https://arxiv.org/abs/1811.01158  
[9] Wang et al. *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*. ICLR 2019 submission / arXiv 2018. https://arxiv.org/abs/1804.07461  
[10] Wang et al. *SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems*. NeurIPS 2019. https://arxiv.org/abs/1905.00537  
[11] Cobbe et al. *Training Verifiers to Solve Math Word Problems*. arXiv 2021. https://arxiv.org/abs/2110.14168  
[12] Hendrycks et al. *Measuring Massive Multitask Language Understanding*. ICLR 2021. https://arxiv.org/abs/2009.03300  
