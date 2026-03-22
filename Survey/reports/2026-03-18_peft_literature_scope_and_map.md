# Sparse, Tensorized, and Interpretable PEFT Literature Map

Date: 2026-03-18

## Scope

This survey is restricted to the literature needed to answer one project-defining question:

Can a sparse tensorized adapter for LLM fine-tuning still offer a publication-level novelty claim once the existing LoRA, tensorized PEFT, sparse PEFT, and interpretability-adjacent literature are taken into account?

The search boundary is intentionally narrow:

1. LoRA-style parameter-efficient fine-tuning as the baseline family.
2. Tensorized PEFT methods that replace matrix low-rank updates with higher-order tensor parameterizations.
3. Sparse PEFT methods that reduce trainable adaptation through rank sparsification, sparse masks, or sparse update schedules.
4. Interpretability-adjacent work that either studies adapter structure directly or uses sparse factor models to improve mechanistic interpretability.

The survey excludes:

1. Generic adapter tuning papers without direct relevance to low-rank, tensor, or sparse structure.
2. Pure model compression papers that do not concern fine-tuning or adaptation.
3. General mechanistic interpretability work that does not connect to adapters or sparse adaptation.

## Core Literature Clusters

| Cluster | Representative papers | Main mechanism | Relevance to project |
| --- | --- | --- | --- |
| Low-rank PEFT baseline | LoRA (Hu et al., 2021) | Matrix low-rank update per target weight | Baseline method and baseline efficiency claim |
| Sparse low-rank PEFT | SoRA (Ding et al., EMNLP 2023), RoseLoRA (Wang et al., EMNLP 2024) | Sparsify rank components or row/column structure within LoRA-style updates | Shows sparse adaptation around LoRA is already established |
| Sparse fine-tuning for LLMs | Scaling Sparse Fine-Tuning to LLMs / SpIEL (Ansell et al., 2024), SPP (Lu et al., ICML 2024) | Sparse update or sparsity-preserved fine-tuning of sparse models | Shows efficiency via sparsity alone is already a strong line |
| Tensorized PEFT | LoTR (Bershatsky et al., 2024), LoRETTA (Yang et al., NAACL 2024), LoRTA (Hounie et al., 2024), DoTA (2024) | Tensor decomposition of adapter updates or tensorized weight reparameterization | Shows tensorized PEFT is already an active line |
| Interpretability-adjacent sparse adaptation | Boosted Sparse and Low-Rank Tensor Regression (He et al., NeurIPS 2018), Low-Rank Adapting Models for Sparse Autoencoders (Chen et al., ICML 2025) | Sparse unit-rank pathways or LoRA-based adaptation around sparse feature models | Most relevant conceptual bridge for an interpretable sparse tensor adapter |

## Paper Notes

### 1. LoRA: Low-Rank Adaptation of Large Language Models

- Citation: Hu et al., 2021. arXiv:2106.09685.
- Key idea: freeze pretrained weights and learn low-rank updates.
- What matters here: LoRA is the default PEFT baseline and establishes the standard efficiency narrative. Any proposed method must beat or clearly complement LoRA in either parameter count, runtime, or analysis value.
- Source: https://arxiv.org/abs/2106.09685

### 2. Boosted Sparse and Low-Rank Tensor Regression

- Citation: He et al., NeurIPS 2018. arXiv:1811.01158.
- Key idea: represent the coefficient tensor as sparse unit-rank CP components.
- Most important conceptual statement from the paper: sparse unit-rank tensor components correspond to a few distinct pathways, each involving only subsets of feature dimensions.
- What matters here: this is the strongest source for the project's interpretability intuition, not for an LLM-ready method. Its contribution is pathway sparsity plus interpretability, not adapter engineering.
- Source: https://arxiv.org/abs/1811.01158

### 3. SoRA: Sparse Low-rank Adaptation of Pre-trained Language Models

- Citation: Ding et al., EMNLP 2023.
- Key idea: introduce sparse gates over LoRA rank components and prune zeroed rank blocks at inference time.
- What matters here: this already occupies part of the sparse-plus-low-rank design space. A new method cannot claim novelty just by adding sparsity to LoRA.
- Source: https://aclanthology.org/2023.emnlp-main.252/

### 4. LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models

- Citation: Yang et al., NAACL 2024.
- Key idea: use tensor-train decomposition for ultra-low-parameter LLM fine-tuning.
- What matters here: tensorized adapters for LLMs are already established. LoRETTA is especially important because it argues for much lower parameter budgets and reports strong efficiency behavior.
- Source: https://aclanthology.org/2024.naacl-long.174/

### 5. LoTR: Low Tensor Rank Weight Adaptation

- Citation: Bershatsky et al., 2024. arXiv:2402.01376.
- Key idea: express the adaptation update as a tensor decomposition instead of plain matrix factorization.
- What matters here: direct evidence that tensorized LoRA generalizations are already active prior art.
- Source: https://arxiv.org/abs/2402.01376

### 6. LoRTA: Low Rank Tensor Adaptation of Large Language Models

- Citation: Hounie et al., 2024. arXiv:2410.04060.
- Key idea: use higher-order CP decomposition to represent model updates compactly and flexibly.
- What matters here: this is the closest prior art to the teacher-suggested direction because it already brings CP-style tensorization into LLM PEFT. Any project using CP-like tensor structure must differentiate itself from LoRTA very clearly.
- Source: https://arxiv.org/abs/2410.04060

### 7. Scaling Sparse Fine-Tuning to Large Language Models

- Citation: Ansell et al., 2024. arXiv:2401.16405.
- Key idea: scalable sparse fine-tuning methods for LLMs, including compatibility with quantization and efficient optimizers.
- What matters here: sparse PEFT is no longer a niche baseline. A new method should not claim sparsity-enabled scaling as its only novelty.
- Source: https://arxiv.org/abs/2401.16405

### 8. SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models

- Citation: Lu et al., ICML 2024. arXiv:2405.16057.
- Key idea: preserve existing sparse structure in pruned LLMs during PEFT.
- What matters here: if the project argues about acceleration on sparse LLMs, SPP is a required baseline or at least a required discussion point.
- Source: https://arxiv.org/abs/2405.16057

### 9. Low-Rank Adapting Models for Sparse Autoencoders

- Citation: Chen et al., ICML 2025.
- Key idea: use LoRA to adapt a model around a pretrained sparse autoencoder and improve the interpretability-performance trade-off.
- What matters here: this is not a tensorized adapter paper, but it is evidence that low-rank adaptation has already been used in service of interpretability rather than only efficiency.
- Source: https://proceedings.mlr.press/v267/chen25r.html

## Cluster-Level Conclusions

### What is already covered by prior work

1. Low-rank PEFT for LLMs is mature.
2. Tensorized PEFT for LLMs is already a recognized design family.
3. Sparse PEFT for LLMs is already a recognized design family.
4. Sparse plus low-rank PEFT already exists in the form of SoRA and related work.

### What appears less covered

1. Treating sparse tensor components as explicitly interpretable task pathways is not yet the dominant framing in LLM PEFT papers.
2. Adapter-level interpretability metrics are much less standardized than efficiency or downstream accuracy metrics.
3. Existing tensorized PEFT papers mostly optimize parameter efficiency and compression, not pathway-level explanation.

## Refined Novelty Boundary

The following claims are weak and should not be used as the main contribution:

1. "We apply tensor decomposition to LoRA."
2. "We add sparsity to LoRA."
3. "We use CP decomposition for LLM adapters."

The following claim still appears viable:

Design a sparse tensor adapter whose components are intended to function as analyzable task pathways, then evaluate those pathways with explicit adapter-level interpretability diagnostics in addition to standard PEFT efficiency metrics.

## Implications for the Next Stage

The ideation stage should compare at least three candidate directions:

1. CP-style sparse pathway adapters with component-level sparsity and ablation analysis.
2. Tensorized adapters with structured sparsity aligned to layers, heads, or projection groups for better interpretability and possible acceleration.
3. Interpretability-first PEFT, where the main novelty is the evaluation protocol plus a minimally new sparse tensor parameterization.

The project should treat the following papers as mandatory discussion baselines:

1. LoRA.
2. SoRA.
3. LoRETTA.
4. LoTR.
5. LoRTA.
6. Scaling Sparse Fine-Tuning to LLMs.
7. SPP.

## Bottom-Line Assessment

The teacher's intuition is still usable, but only after refinement.

Directly implementing the 2018 sparse low-rank tensor regression idea in an LLM adapter is not, by itself, a strong novelty claim in 2026. The viable path is to turn the "sparse pathways are interpretable" intuition into a distinct adapter design and an explicit interpretability evaluation protocol that existing sparse or tensorized PEFT papers do not already provide.
