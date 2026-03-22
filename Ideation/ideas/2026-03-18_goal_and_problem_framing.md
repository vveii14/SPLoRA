# Refined Research Goal and Problem Framing

Date: 2026-03-18

## Refined Research Goal

Develop a parameter-efficient LLM adapter that combines tensorized structure with explicit structured sparsity so that the learned adaptation decomposes into a small number of analyzable components, and test whether those components retain LoRA-level utility while improving adapter-level interpretability under predefined diagnostics.

Operationally, the project should aim to satisfy four conditions:

1. downstream quality remains competitive with LoRA at comparable adaptation budgets,
2. trainable parameter count and training cost remain within practical PEFT ranges,
3. learned components are sparse and structurally concentrated enough to analyze,
4. component behavior is stable enough across ablations or random seeds to support interpretability claims.

## Refined Problem Framing

The project should not be framed as a generic extension of LoRA with tensors or sparsity. That framing is too weak because low-rank PEFT, sparse PEFT, and tensorized PEFT are all already established in the literature.

The defensible problem is narrower:

Existing PEFT methods are optimized primarily for efficiency and quality, but they do not usually make the learned adapter decomposition itself a first-class scientific object. As a result, adapter components are often compact but not clearly interpretable. The motivating hypothesis from sparse tensor regression is that if the update is decomposed into structured sparse tensor components, then each component may behave like a task pathway over a restricted subset of architectural dimensions. If true, this would provide a better trade-off between efficiency and analyzability than standard LoRA-style low-rank adapters.

This framing imposes three hard constraints on the eventual method:

1. it must be clearly different from LoRTA-style CP tensorization alone,
2. it must be clearly different from SoRA-style sparsified low-rank adaptation alone,
3. it must define pathway-level diagnostics before experimentation rather than treating interpretability as a post-hoc visualization exercise.

## Falsifiable Core Claim

A sparse tensorized adapter can produce a smaller set of more interpretable adaptation components than standard LoRA-style adapters, without incurring an unacceptable drop in downstream performance or efficiency.

## Immediate Consequence for the Next Ideation Step

Method ideation should compare only candidate designs that explicitly target pathway structure, not arbitrary tensor or sparse variants.
