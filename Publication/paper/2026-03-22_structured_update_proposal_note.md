# Structured Update Proposal Note

## 1. What This Proposal Should Actually Be About

This proposal should not be written as a loose stack of ingredients:

\[
\text{LoRA} + \text{rank-1 decomposition} + \text{regularization} + \text{tensor decomposition}.
\]

That formulation is too weak because each piece already has direct precedent in the literature [1][2][3][4][5][6][7]. The proposal needs a tighter scientific center. The strongest formulation available after the survey is:

\[
\text{structured parameter updates should be made explicit, regularized for identifiability, and solved at the component level.}
\]

This changes the object of interest. Instead of treating LoRA as merely a low-rank parameter-saving trick, the proposal treats the adapter update itself as a structured object whose internal decomposition should be visible, constrained, and analyzable.

## 2. Level 1: Reparameterize the Update

The starting point is standard LoRA:

\[
W = W_0 + \Delta W, \qquad \Delta W = BA,
\]
with
\[
B \in \mathbb{R}^{d_{\text{out}} \times r}, \qquad
A \in \mathbb{R}^{r \times d_{\text{in}}}.
\]

Because $B=[b_1,\dots,b_r]$ and $A=[a_1^\top;\dots;a_r^\top]$, the update can be rewritten as

\[
\Delta W = BA = \sum_{k=1}^{r} b_k a_k^\top.
\]

This identity is already implicit in LoRA [1], and later work such as AROMA shows that rank structure itself can be made algorithmically explicit [2]. So the proposal cannot claim novelty merely by exposing the sum. The actual move is more specific:

\[
\Delta W = \sum_{k=1}^{K} \sigma_k \hat{u}_k \hat{v}_k^\top,
\qquad
\|\hat{u}_k\|_2 = \|\hat{v}_k\|_2 = 1.
\]

Here, $\sigma_k$ is the component strength and $(\hat{u}_k,\hat{v}_k)$ represent a normalized rank-1 pathway. This is not just a re-expression of LoRA. It changes the primitive modeling unit from factor blocks $(A,B)$ to explicit pathway objects $\{\sigma_k,\hat{u}_k,\hat{v}_k\}$.

Once the update is written in this form, the following questions become natural:

- which pathways are active?
- which pathways dominate the update norm?
- which pathways survive pruning or ablation?
- which pathways are stable across random seeds?

That is the main value of Level 1. The update decomposition becomes a first-class scientific object.

## 3. Level 2: Regularize the Pathways

With standard factor-first LoRA, the regularized objective is usually written as

\[
\mathcal{L}
=
\mathcal{L}_{\text{task}}(W_0+\Delta W)
 \lambda \mathcal{R},
\]
with penalties of the form
\[
\mathcal{R}_{\text{add}}
=
\|A\| + \|B\|
\]
or, in component language,
\[
\mathcal{R}_{\text{add}}
=
\sum_{k=1}^{K}
\left(
\alpha \|\hat{u}_k\|_p + \beta \|\hat{v}_k\|_q
\right).
\]

This is enough to shrink parameters, but it is not aligned with the pathway view. If the true object is an outer-product component, then complexity should be measured at the component level, not only at the factor level.

The simplest pathway regularizer is amplitude sparsity:

\[
\mathcal{R}_{\sigma} = \sum_{k=1}^{K} |\sigma_k|.
\]

This already changes the inductive bias from ``shrink all factors'' to ``use as few pathways as possible.'' A stronger option is a multiplicative pathway penalty:

\[
\mathcal{R}_{\text{mult}}
=
\sum_{k=1}^{K}
\|\hat{u}_k\|_1 \|\hat{v}_k\|_1,
\]
or
\[
\mathcal{R}_{\text{mult-scale}}
=
\sum_{k=1}^{K}
|\sigma_k| \|\hat{u}_k\|_1 \|\hat{v}_k\|_1.
\]

This is the part of the proposal most closely connected to boosted sparse and low-rank tensor regression [8]. That paper is not a LoRA paper, but it provides the right structural intuition: when the model is built from outer-product components, the regularizer should reflect outer-product structure rather than only separate factor magnitudes.

This is also where the proposal differs from existing regularized LoRA methods such as SoRA [3] and SPP [4]. Those papers already show that regularization and sparsity matter for LoRA. The claim here is narrower:

\[
\text{the regularizer should help identify pathways, not only shrink parameters.}
\]

## 4. Level 3: Solve at the Component Level

After Levels 1 and 2, the optimization problem becomes

\[
\min_{\{\sigma_k,\hat{u}_k,\hat{v}_k\}_{k=1}^{K}}
\mathcal{L}_{\text{task}}
\left(
W_0+\sum_{k=1}^{K}\sigma_k \hat{u}_k \hat{v}_k^\top
\right)
 \lambda \mathcal{R}\big(\{\sigma_k,\hat{u}_k,\hat{v}_k\}\big).
\]

At this point, continuing to think only in terms of monolithic factor optimization is no longer natural. The decomposition

\[
\Delta W^{(t)} = \sum_{k=1}^{K} \Delta W_k^{(t)}, \qquad
\Delta W_k^{(t)} = \sigma_k^{(t)} \hat{u}_k^{(t)} \hat{v}_k^{(t)\top}
\]
exposes an explicit modular update structure. A solver can then be defined over components:

\[
(\sigma_k^{(t+1)},\hat{u}_k^{(t+1)},\hat{v}_k^{(t+1)})
=
\mathcal{S}_k(\nabla \mathcal{L}, \mathcal{R}, \Delta W^{(t)}).
\]

The proposal should be careful here. The claim is not that a new solver is automatically better because it is new. The claim is that once the model is written in explicit pathways, a component-aware solver becomes a direct computational consequence of the model structure. That is why Level 3 should be presented as solver redesign induced by the reformulation, not as an isolated engineering optimization.

## 5. Matrix to Tensor Is an Extension, Not the Main Novelty

The matrix case is not the whole story, but it must be the first story. Once the structured-update view is accepted, the natural extension is

\[
\Delta \mathcal{W}
=
\sum_{k=1}^{K}
\sigma_k
\hat{u}^{(1)}_k \otimes \hat{u}^{(2)}_k \otimes \cdots \otimes \hat{u}^{(M)}_k.
\]

This is where tensorization enters. But tensorization itself is not a novelty claim anymore, because LoRETTA [5], LoTR [6], and LoRTA [7] already occupy that space. Tensorization should therefore be used to support a narrower statement:

\[
\text{if pathways are the basic units, tensorization aligns those pathways with architectural modes.}
\]

That is a more defensible claim than ``we use tensor LoRA.''

## 6. What Existing Work Has Already Done

| Topic | Existing papers | What is already done | What remains open |
| --- | --- | --- | --- |
| Low-rank PEFT baseline | LoRA [1] | Low-rank update parameterization is established | Not novel |
| Rank-1 / componentized view | AROMA [2] | Rank structure can be made explicit | Components are not yet the central scientific object |
| Regularized LoRA | SoRA [3], SPP [4] | Sparsity and regularization over LoRA already exist | Pathway identifiability is not the main stated target |
| Tensorized LoRA | LoRETTA [5], LoTR [6], LoRTA [7] | Tensorization is established | Tensorization cannot stand alone as the novelty claim |
| Structure-induced regularization intuition | Boosted Sparse and Low-Rank Tensor Regression [8] | Multiplicative structural regularization is conceptually supported | Not yet translated cleanly into a pathway-first LoRA proposal |

## 7. What This Proposal Should Claim

The proposal should not say:

> We propose LoRA with rank-1 decomposition, regularization, and tensor decomposition.

It should say something closer to:

> We do not claim novelty from rank-1 decomposition, regularization, or tensorization in isolation. We claim novelty from treating adapter components as explicit structured pathways, regularizing them for identifiability at the component level, and evaluating them as first-class interpretable objects. Tensorization then serves as the natural generalization of the same pathway view to higher-order parameter structure.

That version is narrower, but it is also much more likely to survive contact with the literature.

## 8. Immediate Experimental Consequence

The first implementation path should therefore be conservative:

1. stay in the matrix setting,
2. use explicit rank-1 pathway decomposition,
3. start with pathway amplitude sparsity over $\sigma_k$,
4. then test multiplicative pathway regularization,
5. treat tensorization as a second-stage extension after the matrix-core logic is validated.

This ordering cleanly separates the three internal claims:

- update modeling,
- regularization,
- solver design.

It also gives the project a much cleaner ablation story.

## 9. Candidate Code and Dataset Support

Public code references already acquired in this workspace:

| Method | Local repo |
| --- | --- |
| LoTR [6] | `Experiment/code_references/lotr` |
| LoRTA [7] | `Experiment/code_references/lorta` |
| LoRETTA [5] | `Experiment/code_references/loretta` |
| SPP [4] | `Experiment/code_references/spp` |

Suggested benchmark ladder for later experiment design:

| Stage | Dataset | Citation |
| --- | --- | --- |
| Small validation | GLUE | [9] |
| Stronger classification | SuperGLUE | [10] |
| Reasoning transfer | GSM8K | [11] |
| Broad capability check | MMLU | [12] |

## 10. Environment Notes

Two requested skill executions were blocked by the current environment:

- `gemini-deep-research` could not run because `GEMINI_API_KEY` is not set.
- `inno-figure-gen` could not run because both `uv` and `GEMINI_API_KEY` are unavailable.

This does not affect the core literature conclusions, but it does mean no AI-generated figure was produced in this pass.

## References

[1] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv 2021. https://arxiv.org/abs/2106.09685  
[2] Lu et al. *AROMA: Adaptive Rank Reduction of Low-Rank Adaptation for Large Language Models*. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.170/  
[3] Ding et al. *Sparse Low-rank Adaptation of Pre-trained Language Models*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.252/  
[4] Lu et al. *SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models*. arXiv 2024. https://arxiv.org/abs/2405.16057  
[5] Yang et al. *LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models*. NAACL 2024. https://aclanthology.org/2024.naacl-long.174/  
[6] Bershatsky et al. *LoTR: Low Tensor Rank Weight Adaptation*. arXiv 2024. https://arxiv.org/abs/2402.01376  
[7] Hounie et al. *LoRTA: Low Rank Tensor Adaptation of Large Language Models*. arXiv 2024. https://arxiv.org/abs/2410.04060  
[8] He et al. *Boosted Sparse and Low-Rank Tensor Regression*. NeurIPS 2018. https://arxiv.org/abs/1811.01158  
[9] Wang et al. *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*. https://arxiv.org/abs/1804.07461  
[10] Wang et al. *SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems*. https://arxiv.org/abs/1905.00537  
[11] Cobbe et al. *Training Verifiers to Solve Math Word Problems*. https://arxiv.org/abs/2110.14168  
[12] Hendrycks et al. *Measuring Massive Multitask Language Understanding*. https://arxiv.org/abs/2009.03300  
