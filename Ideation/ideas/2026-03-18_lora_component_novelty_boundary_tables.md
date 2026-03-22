# LoRA Component Novelty Boundary Tables

Date: 2026-03-18

Purpose: summarize which parts of the proposed method chain already have direct precedent in the LoRA and PEFT literature, and identify where a defensible contribution may still remain for task 9.

## Step 1: Standard LoRA

| Item | Summary |
| --- | --- |
| Core form | \(\Delta W = BA\) |
| Role | Baseline low-rank PEFT formulation |
| Representative work | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) |
| Maturity | Established and standard |
| Implication for this project | Cannot be claimed as novelty; must remain a baseline |

## Step 2: LoRA with Rank-1 or Componentized Decomposition

| Item | Summary |
| --- | --- |
| Core form | \(\Delta W = \sum_{r=1}^{R} b_r a_r^\top\) |
| Role | Makes the low-rank update explicit as a sum of rank-1 components or pathways |
| Prior precedent | Yes |
| Representative directions | AROMA-style progressive rank-one construction; analysis work such as *Rank-1 LoRAs Encode Interpretable Reasoning Signals* |
| Why it still matters | Useful for component pruning, pathway analysis, and interpretability-oriented diagnostics |
| Novelty risk | Too weak if presented alone |

## Step 3: LoRA with Regularization

| Item | Summary |
| --- | --- |
| Common forms | Sparsity penalties, gate regularization, Fisher regularization, task-specific regularization |
| Prior precedent | Yes, with several variants |
| Representative directions | SoRA, FR-LoRA, contrastive regularization variants, other sparse or stability-oriented LoRA methods |
| What remains less explored | Factorization-induced multiplicative or component-level regularization in the style of sparse low-rank tensor regression |
| Opportunity for this project | A pathway-level multiplicative regularizer may still differentiate the method if tied to explicit component identifiability |
| Novelty risk | Generic regularization over LoRA is not new |

## Step 4: LoRA with Tensor Decomposition

| Item | Summary |
| --- | --- |
| Core idea | Lift the update from a matrix parameterization to a higher-order tensor parameterization and decompose it |
| Common decompositions | CP, Tucker, Tensor-Train, Tensor-Ring or related multilinear forms |
| Prior precedent | Yes |
| Representative directions | LoRETTA, LoTR, LoRTA |
| Why it matters | Can exploit multiway structure such as head, group, layer, or projection modes |
| Novelty risk | Tensorized LoRA alone is already a populated sub-literature |

## Combined Assessment

| Combination | Literature status | Judgment |
| --- | --- | --- |
| LoRA | Mature | Baseline only |
| LoRA + rank-1 decomposition | Already explored | Not sufficient as novelty |
| LoRA + regularization | Already explored | Not sufficient as novelty |
| LoRA + tensor decomposition | Already explored | Not sufficient as novelty |
| LoRA + rank-1 + regularization + tensorization | Constituents all have precedent | Direct composition is likely to be judged incremental |
| Pathway-first rank-1 or tensorized components + multiplicative regularization + interpretability-first evaluation | No single mature directly matching representative was identified in the current survey | Most plausible novelty zone |

## Working Claim

The project should not claim novelty from rank-1 decomposition, regularization, or tensorization in isolation. The defensible claim is narrower: adapter components should be made explicit as structured task pathways, regularized for identifiability at the component level, and evaluated as first-class interpretable objects.

## Implication for Task 9

This table set supports the selection of the first research direction by narrowing the viable claim space. The strongest remaining direction is not generic sparse tensorized LoRA, but a pathway-first structured-update framing in which:

1. rank-1 or tensorized components are explicit primitive units,
2. regularization acts on pathways rather than only on factors,
3. tensorization is justified as alignment with architectural modes rather than as a standalone novelty claim, and
4. interpretability evaluation is built into the method definition rather than added post hoc.
