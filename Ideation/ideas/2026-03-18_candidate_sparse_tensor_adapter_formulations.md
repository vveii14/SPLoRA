# Candidate Sparse Tensor Adapter Formulations

Date: 2026-03-18

## Objective

Generate candidate method families that satisfy the survey constraint:

do not propose a generic tensorized or sparse LoRA variant; propose a method that explicitly targets interpretable sparse pathways and can be differentiated from LoRTA, LoRETTA, SoRA, and broader sparse PEFT baselines.

## Evaluation Axes for Ideation

Each candidate is judged on four axes:

1. novelty leverage,
2. implementation complexity,
3. interpretability signal,
4. fit for a first round of experiments on 4x A100.

## Candidate A: CP-Path

### Core idea

Represent each adapter update as a sum of a small number of CP-style components, but impose structured sparsity on the factors so that each component is active only on a restricted subset of layers, heads, channel groups, or projection groups. The main scientific object is the component itself, not just the aggregate update.

### Why it may work

1. It preserves the most direct connection to the pathway intuition from boosted sparse and low-rank tensor regression.
2. Component ablation is naturally defined because each pathway is already a separate factorized unit.
3. It offers a clean story for pathway sparsity, concentration, and stability.

### Main novelty hook

Not "CP for adapters," but "CP components as sparse task pathways with explicit interpretability diagnostics."

### Main reviewer risk

This is closest to LoRTA. If the method looks like CP decomposition plus an L1 or group-lasso penalty, the paper may read as a light extension.

### Mitigation

The sparsity structure must be meaningfully tied to pathway semantics, for example:

1. sparsity over architectural groups rather than raw entries,
2. regularizers that encourage component separation or concentration,
3. diagnostics that show different components control different behaviors.

### First-round feasibility

High. This is probably the easiest family to implement first.

## Candidate B: Group-Structured Tensor Paths

### Core idea

Use a tensorized adapter, but define sparsity over interpretable architectural groups from the start, such as:

1. attention heads,
2. MLP channel groups,
3. query or value projection groups,
4. layer blocks.

Instead of asking the model to discover arbitrary sparse structure, ask it to discover sparse structure over semantically meaningful groups.

### Why it may work

1. Structured groups are easier to interpret than arbitrary sparse factors.
2. Structured sparsity is more compatible with acceleration arguments than unstructured sparsity.
3. This gives a stronger differentiator from plain CP-style tensorization.

### Main novelty hook

The contribution is a pathway-aware structural prior: adapter components are sparse over architectural groups that are already meaningful to transformer analysis.

### Main reviewer risk

If the grouping is arbitrary or heuristic, reviewers may see it as hand-designed bias without enough justification.

### Mitigation

Choose groups that already matter in transformer analysis and can be measured clearly in ablations. For example, layer groups and head groups are easier to defend than fully custom partitions.

### First-round feasibility

Medium to high. Slightly more engineering than Candidate A, but still realistic.

## Candidate C: Mixture-of-Pathways Adapter

### Core idea

Learn a small bank of sparse tensor components and let the model route examples or tokens through a subset of them, similar in spirit to a lightweight adapter-level mixture of experts. The emphasis is on interpretability of which pathway bank is selected and when.

### Why it may work

1. Explicit routing could produce clearer pathway specialization.
2. It creates a direct link between data subsets and pathway usage.
3. Routing statistics provide an additional interpretability signal.

### Main novelty hook

Pathways are not just decomposition units; they are conditionally selected adaptation experts.

### Main reviewer risk

This may drift too far toward MoE-style routing or X-LoRA-style adapter mixtures and lose the simplicity advantage of PEFT.

### Mitigation

Keep routing lightweight and secondary. The paper should remain about sparse tensor pathways, not about building a full adapter MoE system.

### First-round feasibility

Medium to low. Interesting, but likely too ambitious as the first implementation.

## Candidate D: Interpretability-First Minimal Adapter

### Core idea

Keep the adapter design only modestly new, for example tensorized plus structured sparsity over a small number of components, but make the stronger contribution the evaluation protocol:

1. component concentration,
2. ablation impact,
3. cross-seed stability,
4. alignment with task clusters or data strata,
5. architectural localization of pathways.

### Why it may work

1. It avoids overengineering the method before the interpretability question is validated.
2. It reduces implementation risk.
3. It could still produce a useful paper if the evaluation framework is genuinely sharp and the method exposes analyzable components.

### Main novelty hook

The scientific contribution is the combination of a minimally pathway-aware sparse tensor adapter with an explicit benchmark for adapter-level interpretability.

### Main reviewer risk

If the adapter itself looks too incremental, reviewers may say the evaluation protocol belongs in an analysis paper rather than a methods paper.

### Mitigation

The minimal adapter must still include at least one nontrivial mechanism, such as structured sparsity over semantically meaningful groups or an objective that encourages pathway separation.

### First-round feasibility

Very high. This is the safest starting point if time pressure dominates.

## Candidate E: Hierarchical Tensor Paths

### Core idea

Use a two-level decomposition:

1. coarse sparse pathway allocation across layers or modules,
2. fine low-rank or tensorized adaptation inside each selected region.

This is a hierarchical pathway model rather than a flat set of tensor components.

### Why it may work

1. Coarse-to-fine structure may produce clearer localization.
2. It makes the adapter easier to visualize and interpret.
3. It naturally supports analyses such as "which parts of the network changed for which task."

### Main novelty hook

Hierarchical pathway structure, not merely sparse tensor factorization.

### Main reviewer risk

This may become too complex before the base hypothesis is validated.

### Mitigation

Only consider this if a simpler flat-pathway model already shows promise.

### First-round feasibility

Low for first implementation. Better as a second-wave direction.

## Comparative Ranking for First Implementation

### Best balance of novelty and feasibility

1. Candidate B: Group-Structured Tensor Paths
2. Candidate A: CP-Path
3. Candidate D: Interpretability-First Minimal Adapter

### Highest upside but higher risk

1. Candidate C: Mixture-of-Pathways Adapter
2. Candidate E: Hierarchical Tensor Paths

## Recommended Shortlist for Task 9

The next narrowing step should compare these three:

1. Candidate B as the most differentiated methods direction.
2. Candidate A as the simplest pathway-faithful baseline direction.
3. Candidate D as the safest fallback if implementation complexity becomes the bottleneck.

## Working Recommendation

At this point, the strongest practical direction is:

start from Group-Structured Tensor Paths, but keep the first prototype close enough to CP-Path that implementation remains manageable. In other words, build a sparse tensor adapter whose sparsity is imposed over meaningful architectural groups, and evaluate whether the resulting components are more interpretable than standard LoRA-style low-rank adapters.

This recommendation is stronger than pure CP-plus-sparsity because it gives a cleaner novelty hook and a more defensible interpretability story.
