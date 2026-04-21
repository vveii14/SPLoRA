# A Structured Survey of Low-Rank Adaptation (LoRA) for Large Language Models: Design Axes, Intellectual Trajectories, and Open Problems

**Scope.** This document surveys the Low-Rank Adaptation (LoRA) literature from its introduction in 2021 through early 2026, with emphasis on work at NeurIPS / ICLR / ICML / ACL venues. We cover roughly 80 papers and organize them along six orthogonal design axes defined by which component of the base LoRA formulation they modify. The aim is not encyclopedic coverage, but a structured map of the design space that makes it possible to (i) locate any new proposal within the existing landscape and (ii) identify mechanistically distinct open problems that remain unaddressed.

**Audience.** Researchers entering or working in parameter-efficient fine-tuning (PEFT) who want a condensed analytical view rather than a chronology of individual methods.

---

## 1. Introduction and Design Space

LoRA, introduced by Hu et al. (2106.09685), adapts a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ by learning a low-rank additive perturbation:

$$W = W_0 + \Delta W, \qquad \Delta W = B A, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k).$$

During training $W_0$ is frozen and only $A, B$ are updated. At inference $BA$ can be merged back into $W_0$, yielding zero latency overhead. The original motivation — drawn from the intrinsic-dimensionality results of Aghajanyan et al. (2012.13255) — is that fine-tuning updates empirically live in a low-dimensional subspace, so a rank-$r$ parameterization with $r \sim 8$ suffices for most downstream tasks.

The resulting design space is narrower than full fine-tuning, but each of the structural choices made in defining LoRA is itself a **modifiable design decision**. Every subsequent variant in the literature can be understood as a modification to one of the following axes:

| Axis | Modifiable component | Representative question |
|---|---|---|
| 1. **Structure / decomposition** | The functional form of $\Delta W$ | Why a bilinear product? Why not a tensor, Kronecker, Hadamard, or spectral representation? |
| 2. **Initialization & rank allocation** | How $A, B$ are initialized and how $r$ is chosen per layer | Can data signal — gradients, activations, Fisher — guide initialization and rank budget? |
| 3. **Optimization & geometry** | The update rule for $(A, B)$ | Does AdamW suffice, or can curvature, manifold structure, or second-order information be exploited? |
| 4. **Sparsity, merging, mixture** | How multiple adapters or sparse patterns are composed | How should multiple LoRAs be combined without destructive interference? |
| 5. **Function / application** | What LoRA is *used for* beyond task adaptation | Can LoRA enable interpretable surgery, unlearning, or safety alignment? |
| 6. **Theory, quantization, federation** | Formal guarantees and deployment constraints | What can be proven about expressiveness and generalization? How does LoRA interact with quantization and distributed training? |

A persistent source of confusion in the literature — and a recurring pitfall in novelty assessments — is the conflation of **mechanism** with **topic**. Two methods may both operate on, e.g., "the geometry of LoRA updates" while using entirely different mathematical objects (a parameterization metric vs. a loss Fisher). Throughout this survey we distinguish methods by their underlying mechanism rather than their thematic label, since mechanism-level differences determine whether a new proposal is genuinely novel or incrementally overlapping.

The axes above are not mutually exclusive — many papers touch multiple — but most have a primary locus of contribution, which we use for classification.

---

## 2. Axis 1: Structural and Decomposition Variants

The most immediately visible design choice is the *functional form* of $\Delta W$. Bilinear $BA$ is one choice among many; the literature explores three broad families of alternatives.

### 2.1 Bilinear with Reparameterization or Sharing

The first family retains the two-matrix skeleton but modifies what is trainable, what is shared, or how the matrices interact.

**DoRA** (ICML'24, 2402.09353) decomposes each column of $W$ into a magnitude scalar $m$ and a direction vector, training $m$ jointly with a standard $BA$ update on the direction. The motivation is an empirical observation — confirmed in several follow-ups — that vanilla LoRA underfits the magnitude component relative to full fine-tuning, leading to a characteristic learning-rate mismatch between the two.

**VeRA** (ICLR'24, 2310.11454) pushes parameter sharing to an extreme: $A$ and $B$ are **frozen random matrices shared across all layers**, and each layer learns only two diagonal scaling vectors $\Lambda_b, \Lambda_d$, giving $\Delta W = \Lambda_b B \Lambda_d A$. The success of this approach — parameter reduction of $\sim 97\%$ with modest accuracy loss — is in itself an empirical argument that much of LoRA's capacity is redundant, since fixed random bases provide sufficient representational substrate for many tasks.

**LoRA-FA** (2308.03303) and **LoRA-XS** (2405.17604) explore the middle ground: LoRA-FA freezes $A$ alone, reducing parameters by half; LoRA-XS first computes the SVD of $W_0$, freezes the singular bases $U, V$, and trains only a central $r \times r$ matrix $R$ with $\Delta W = U R V^\top$. Both methods ground the low-rank update in properties of $W_0$ itself rather than learning the bases from scratch.

**VB-LoRA** (NeurIPS'24, 2405.15179) introduces a **global vector bank** and learns a sparse top-$k$ admixture over bank vectors to construct each layer's low-rank factors. Formally, each LoRA rank-1 component is expressed as a sparse combination of atoms drawn from a shared dictionary, yielding a "divide-and-share" paradigm with extreme parameter efficiency (0.4% of LoRA parameters on Llama2-13B). This paper is worth singling out: it effectively instantiates the "shared atom dictionary + sparse selector" idea in its most direct form, and therefore constrains the design space available to any subsequent proposal along this axis.

**Structural variants of BA.** MELoRA (2402.17263) stacks multiple independent $B_i A_i$ along a block diagonal, increasing effective rank without additional parameters. MoSLoRA / MiSS (2406.11909) inserts a learnable $r \times r$ mixer between $B$ and $A$: $\Delta W = B M A$, adding $r^2$ parameters for qualitatively richer subspace mixing. ReLoRA (2307.05695) merges $BA$ back into $W$ periodically and reinitializes, recasting training as a temporal sum of low-rank updates — a device primarily useful in pretraining rather than fine-tuning.

### 2.2 Non-Bilinear Structured Parameterizations

A second family abandons the bilinear form entirely.

**Hadamard and Kronecker structures.** LoHa / FedPara (2108.06098) parameterizes $\Delta W = (B_1 A_1) \odot (B_2 A_2)$, where the Hadamard product's rank multiplicativity lifts the effective rank above $r$ at the cost of doubling parameters. LoKr (2309.14859) uses $\Delta W = A \otimes B$, a Kronecker factorization that yields extreme parameter efficiency but rigid shape constraints.

**Spectral representations.** FourierFT (ICML'24, 2405.03003) departs entirely from the low-rank assumption. It parameterizes $\Delta W$ as the inverse 2D DFT of a sparse set of learned spectral coefficients $S$. The resulting perturbation is neither low-rank nor spatially sparse, but spectrally sparse. This is a notable conceptual break: it demonstrates that "parameter efficiency" and "low rank" are not equivalent, and that sparsity in alternative bases can be competitive.

**Orthogonal (multiplicative) updates.** OFT (NeurIPS'23, 2306.07280), BOFT (ICLR'24, 2311.06243), and HRA (NeurIPS'24, 2405.17484) replace additive $\Delta W$ with a multiplicative orthogonal rotation: $W' = R W_0$ where $R$ is a structured orthogonal matrix (block-diagonal Cayley, butterfly-factored, or Householder-reflector chain, respectively). The shared motivation is preservation of $W_0$'s spectrum during adaptation, which provides some protection against the "intruder singular direction" phenomenon identified by Shuttleworth et al. (2410.21228); see §7.1. The trade-off is that orthogonal constraints limit magnitude changes, which can hurt expressiveness for tasks requiring substantial re-weighting.

### 2.3 Tensor Decomposition Families

The third family treats the collection of LoRA updates across layers and attention heads as a single high-order tensor and applies tensor decompositions for joint compression.

**LoTR** (2402.01376) applies Tensor-Train decomposition across the layer dimension, sharing TT cores across layers. **LoRTA** (NeurIPS'24, 2410.04060) and **TensLoRA / TensorLLM** apply CP or Tucker decomposition to the $(\text{layer}, \text{head}, d, k)$ four-mode tensor formed by stacking all LoRA updates. **SuperLoRA** (2403.11887) provides a unified framework that subsumes LoRA, Kronecker, and Tucker variants as special cases through a common tensor-factoring and projection interface.

The appeal of this family is elegance: a single decomposition expresses cross-layer and cross-head parameter sharing. The empirical reality is that these methods underperform vanilla LoRA on tasks with heterogeneous per-layer demands (e.g., small-sample GLUE tasks like RTE), because forcing all layers through a shared tensor core eliminates capacity for layer-specific specialization. This is a structural limitation, not a tuning failure, and it appears robustly across the family.

### 2.4 Synthesis

Axis 1 is the **most crowded** in the literature. Taxonomically, by 2026 essentially every natural matrix/tensor decomposition has been mapped to $\Delta W$: low-rank (LoRA), rank-lifted (LoHa), shape-structured (LoKr, HRA, BOFT), spectral (FourierFT, SVFT), shared-basis (VeRA, VB-LoRA, Tied-LoRA), tensor-decomposed (LoTR, LoRTA, TensLoRA, SuperLoRA). The remaining unoccupied regions are highly specialized — e.g., conditional Tensor-Train cores gated by input statistics, learned rather than head-aligned block partitions for Tucker — and unlikely to carry a full-paper contribution by themselves.

The more interesting observation is that the structural axis has reached a **diminishing-return regime**. Most recent papers in this family report sub-point GLUE or GSM8K improvements over LoRA at matched parameter counts, and the differentiation among them is increasingly driven by specific deployment scenarios (quantization compatibility, merge-friendliness) rather than by fundamental representational capacity. For researchers with three prior iterations on structural variants (as in our case), continuing along this axis yields progressively thinner contributions.

---

## 3. Axis 2: Initialization and Rank Allocation

If Axis 1 asks *what* $\Delta W$ looks like, Axis 2 asks *how training begins* and *how capacity is distributed*. Work here has proliferated rapidly since late 2023, driven by the empirical observation that data-informed initialization can save thousands of training steps.

### 3.1 Initialization: From Random to Data-Aware

The original LoRA uses Kaiming-random $A$ and zero $B$, ensuring $\Delta W = 0$ at initialization. A wave of subsequent work recognized that this discards prior information — about $W_0$, about gradients, or about activations — that can be used to place $(A, B)$ in a more favorable starting region.

**Gradient-based initialization.** LoRA-GA (NeurIPS'24, 2407.05000) computes the full fine-tuning gradient $\nabla W$ at step 0, takes its top-$r$ SVD, and initializes $(A, B)$ so that $BA$ aligns with this gradient subspace. The effect is that the first few gradient steps no longer need to rotate the low-rank subspace into a useful orientation; it is already there. Empirically this yields $+5.7\%$ on GLUE and $+11.5\%$ on GSM8K over random LoRA. LoRA-One (ICML'25, 2502.01235) provides a convergence-theoretic justification: a single-step gradient SVD suffices asymptotically.

**Weight-SVD initialization.** PiSSA (NeurIPS'24 Spotlight, 2404.02948) takes the opposite perspective: it initializes $(A, B)$ from the **principal singular components** of $W_0$ itself, freezing the residual $W_0 - BA$. The narrative framing is that LoRA learns the most important pretrained directions, while the less important components remain untouched. MiLoRA (2406.09044) inverts this logic — arguing that principal directions *encode* pretrained knowledge and should not be modified — and initializes from the **minor** singular components instead. Neither choice is universally dominant; the preference depends on whether the downstream task benefits from modifying or preserving the principal pretrained structure.

**Activation-based initialization.** EVA (ICLR'25, 2410.07170) recognizes that neither $\nabla W$ nor $W_0$ directly measures which directions of input space are *actually activated* by downstream data. EVA performs incremental SVD on minibatch activations $X$, initializes $A$ from the right singular vectors, and allocates ranks proportionally to explained variance. CorDA (NeurIPS'24, 2406.05223) fuses weight and activation information through $W \cdot C^{1/2}$ where $C = XX^\top$. Both methods address an implicit assumption in gradient-based initialization that the downstream data distribution matches the gradient direction taken at step 0, which is only approximately true.

**Fisher-weighted initialization.** LoftQ (ICLR'24, 2310.08659) and LQ-LoRA (ICLR'24, 2311.12023) use Fisher-weighted approximations primarily to handle quantization error, but the underlying principle — that parameter importance should be measured by second-order loss sensitivity rather than magnitude — extends naturally to initialization. LoRA-DA (2510.24561) formalizes this for the non-quantized case, deriving an asymptotically MSE-optimal initialization subspace from the diagonal Fisher, and currently stands as state-of-the-art among initialization methods. Notably, LoRA-DA uses only the *diagonal* Fisher; the corresponding full-matrix or K-FAC analysis is absent.

**Curvature-based initialization.** A small number of 2025 papers (including Curvature-Guided LoRA) use the Neural Tangent Kernel of the pretrained model as an initialization signal. This is related to but distinct from Fisher-based initialization, since NTK captures representational geometry rather than loss curvature.

**Dynamic initialization via gradient matching.** LoRA-Pro (ICLR'25, 2407.18242) occupies a boundary position between initialization and optimization. Rather than setting $(A, B)$ well once, it solves a Sylvester equation at each step to ensure $B \dot{A} + \dot{B} A$ tracks the full-FT gradient $\nabla W$ as closely as possible. The closed-form solution involves $(B^\top B)^{-1}$ and $(A A^\top)^{-1}$, which coincidentally resemble the Riemannian preconditioner of §4, though the derivation is entirely different.

### 3.2 Rank Allocation: Adaptive Capacity Distribution

A fixed per-layer rank is provably suboptimal — different layers and different attention heads contribute unequally to downstream performance. Allocation methods fall into two classes, distinguished by whether they start from excess capacity and prune, or from minimal capacity and grow.

**Pruning approaches.** AdaLoRA (ICLR'23, 2303.10512) reparameterizes $\Delta W$ in SVD form $P \Lambda Q$ with an orthogonality penalty, and prunes entries of $\Lambda$ by sensitivity score. SoRA (EMNLP'23, 2311.11696) inserts a gate vector $g$ between $A$ and $B$ with proximal $\ell_1$ thresholding, producing continuous rank shrinkage. ALoRA (NAACL'24, 2403.16187) scores individual ranks via SHAP-like ablation — measuring the loss change from zeroing each rank — and reallocates budget to positively-contributing ranks.

**Growing approaches.** IncreLoRA (2308.12043) begins with minimal rank and expands based on importance signals. DyLoRA (EACL'23, 2210.07558) trains nested ranks simultaneously via random truncation, yielding a model that can be deployed at any rank up to its maximum without retraining.

**Structural allocation.** LoRA-Drop (COLING'25, 2402.07721) uses the adapter output norm $\|BAx\|$ as the allocation signal, retaining LoRA on high-output layers and sharing a single adapter elsewhere. ARD-LoRA (2506.18267) introduces continuous per-head $\alpha$ regularizers with $\ell_1$ and total-variation penalties, achieving 99.3% of full-FT performance at 0.32% of parameters on Llama-3.1-70B.

**Optimizer-invariant allocation.** LoRA-RITE (ICLR'25, 2410.20625) addresses a subtle issue: the pair $(A, B)$ has a gauge redundancy under the transformation $(QA, BQ^{-1})$ for any invertible $r \times r$ matrix $Q$. AdamW is not invariant to this reparameterization, introducing spurious instability. LoRA-RITE designs a preconditioner that restores GL($r$)-invariance, yielding $+4$–$7\%$ on GSM8K over Adam.

### 3.3 Redundancy Analysis

A distinct literature examines whether LoRA itself is over-parameterized at typical ranks.

SeLoRA (2506.16787) reparameterizes LoRA in a sparse DCT/Fourier basis and demonstrates that a large fraction of LoRA's expressive capacity is redundant across standard ranks. PrunedLoRA (2510.00192) over-parameterizes during training and applies structured gradient-based pruning with no reactivation, yielding provably robust sparse adapters. LoRI (COLM'25, 2504.07448) makes the redundancy explicit by freezing $A$ as a random projection and training only a sparse $B$ with task-specific non-overlapping supports — a design choice that incidentally solves the multi-task merging problem (§5).

### 3.4 Synthesis

Axis 2 has undergone rapid consolidation. By 2026, essentially every natural signal source — gradient mean (LoRA-GA), weight SVD (PiSSA, MiLoRA), activation SVD (EVA), weight-activation interaction (CorDA), diagonal Fisher (LoRA-DA), NTK curvature (Curvature-Guided LoRA) — has a corresponding initialization method. The remaining gaps are narrower but still plausibly productive:

- The **full-matrix** (non-diagonal) Fisher for initialization has not been systematically explored, despite being a natural generalization of LoRA-DA.
- The **covariance of gradients** $\text{Cov}(\nabla W)$, rather than the mean, has not been used — even though it captures stochastic structure that gradient-mean-based methods discard.
- **Gauge fixing at initialization** (rather than in the optimizer as in LoRA-RITE) is unexplored.
- For rank allocation, **mechanistic attribution signals** (path patching, attribution patching — borrowed from mechanistic interpretability) have not been tried, despite being conceptually distinct from sensitivity-based signals like AdaLoRA or SHAP-like signals like ALoRA.

These constitute secondary but genuine white spaces within an otherwise saturated axis.

---

## 4. Axis 3: Optimization, Curvature, and Geometry

This is the axis where conceptual distinctions matter most, and where the literature is most easily misread. Several methods with superficially similar closed-form update rules arise from entirely different theoretical foundations. We therefore begin with a taxonomy of mechanisms, and only then discuss individual methods.

### 4.1 A Taxonomy of Optimization Mechanisms

Denote the LoRA parameter vector as $\theta = [\text{vec}(A); \text{vec}(B)] \in \mathbb{R}^n$ with $n = r(d_1 + d_2)$. Optimization methods for LoRA can be classified by the origin of their preconditioner:

| Class | Origin | Depends on data/loss? |
|---|---|---|
| (a) Parameterization metric | Geometry of the $BA$ factorization itself | No |
| (b) Manifold constraint | Projection onto a structured manifold (Stiefel, fixed-rank) | No |
| (c) Gradient matching | Closed-form match to a single full-FT gradient | Indirectly (one gradient) |
| (d) Natural gradient / Fisher | Preconditioning by the loss Fisher $F = \mathbb{E}[\nabla L \nabla L^\top]$ | Yes (expectation) |
| (e) Spectral / Muon | Operations on the spectrum of the gradient matrix | No |

Crucially, classes (a) and (d) can produce update rules that *look* structurally similar — both may involve $(B^\top B)^{-1}$-type preconditioners — while representing fundamentally different objects. A preconditioner derived from parameterization geometry is data-independent; a Fisher preconditioner is the expectation of a data-dependent outer product. Treating them as equivalent is the most common error in this literature.

### 4.2 Methods by Class

**Class (a): parameterization metric.** Riemannian Preconditioned LoRA (ICML'24, 2402.02347) updates
$$A \leftarrow A - \eta (B^\top B)^{-1} \nabla_A L, \qquad B \leftarrow B - \eta \nabla_B L (A A^\top)^{-1},$$
with damping. The preconditioner arises from treating $BA$ as a point on the Burer–Monteiro manifold and imposing a metric that is invariant under the gauge transformation $(A, B) \mapsto (QA, BQ^{-1})$. This metric is determined entirely by the current values of $A, B$; it does not depend on the data or the loss landscape. Its purpose is to cancel the intrinsic scaling ambiguity of the factorization, not to exploit curvature.

**Class (b): manifold constraint.** StelLA (NeurIPS'25 Spotlight) and Stiefel-LoRA (2508.17901) constrain $B$ to the Stiefel manifold of orthonormal frames. After each Euclidean gradient step, a QR or Cayley retraction projects $B$ back onto $\text{St}(d, r)$. The motivation, supported empirically, is that AdamW drives the columns of $B$ toward high cosine similarity, collapsing the effective rank; an orthogonality constraint prevents this collapse. OFT and BOFT from Axis 1 are manifold-constrained in a related sense but act multiplicatively on $W_0$ rather than on the LoRA factors.

**Class (c): gradient matching.** LoRA-Pro (ICLR'25, 2407.18242) solves a Sylvester equation at each step so that the first-order expansion $B \dot{A} + \dot{B} A$ best approximates the full-FT gradient $\nabla W$. The closed-form solution contains $(B^\top B)^{-1}$ and $(A A^\top)^{-1}$ factors that resemble Class (a), but the derivation is a single-gradient matching problem, not a metric construction. LoRA+ (2402.12354) is a simplified Class (c) instance: from infinite-width scaling arguments, it prescribes $\eta_B = 16 \eta_A$. AltLoRA (2505.12455) alternates low-rank projections of the gradient and momentum within LoRA subspaces; it is closer to a gradient-approximation method than a curvature method.

**Class (d): natural gradient / Fisher.** This class is surprisingly sparse in the LoRA literature. Laplace-LoRA (2308.13111) uses the Generalized Gauss-Newton matrix to approximate the posterior covariance *after* training for uncertainty quantification, not as an optimizer. Adahessian and Sophia have been applied to LoRA parameters in ablations, but both use *diagonal* Hessian approximations that ignore the rank-$r$ structure of the LoRA subspace. LQ-LoRA, LoRA-DA, and LIBU use Fisher information, but respectively for quantization weighting, initialization, and unlearning direction selection — not as a training-time optimizer preconditioner. We return to this gap in §8.

**Class (e): spectral.** Muon (Jordan et al. 2024) applies Newton-Schulz orthogonalization to the momentum matrix before the update. Several 2025 workshop-level papers apply Muon-style updates to the $B$ factor of LoRA, motivated by the same column-collapse observation that drives Stiefel constraints. Pre-conditioned SGD (PSGD) variants with Lie-group preconditioners also fall here.

### 4.3 Synthesis and the Natural Gradient Gap

The most striking observation from classifying this axis is that **Class (d) — the use of the loss Fisher as a training-time preconditioner in the LoRA rank-$r$ subspace — is effectively unoccupied**. This is counterintuitive: natural gradient is the most principled second-order method for statistical learning, and the full-model intractability that historically prevented its use ($O(d^2)$ storage, $O(d^3)$ inversion) evaporates in the LoRA subspace, where $n = r(d_1 + d_2)$ is on the order of a few thousand per layer.

Let $J \in \mathbb{R}^{d_1 d_2 \times n}$ denote the Jacobian of $\text{vec}(BA)$ with respect to $\theta = [\text{vec}(A); \text{vec}(B)]$. The natural gradient update in the LoRA subspace is
$$\theta_{t+1} = \theta_t - \eta (F_r + \lambda I)^{-1} \nabla_\theta L, \qquad F_r = J^\top F J \in \mathbb{R}^{n \times n}.$$
Using an empirical Fisher $\hat F = \frac{1}{m} \sum g_i g_i^\top$ computed from $m$ minibatch gradients $g_i = \nabla_\theta L_i \in \mathbb{R}^n$, the rank of $\hat F$ is at most $m \ll n$, and the Woodbury identity reduces the required inverse to an $m \times m$ matrix solve. The per-step cost is comparable to an additional forward pass.

To our knowledge, no published method performs this computation as its primary mechanism. The closest neighbors differ in specific, identifiable ways:

- Riemannian Preconditioned LoRA (Class a) uses a data-independent metric.
- LoRA-Pro (Class c) matches a single gradient, not its expectation.
- Laplace-LoRA (Class d) uses Fisher post-hoc for posteriors.
- Sophia and Adahessian (Class d) use diagonal curvature.
- LoRA-DA (2510.24561) uses diagonal Fisher for initialization only.
- Curvature-Guided LoRA uses NTK curvature for initialization, not for training dynamics.

This gap is, to our assessment, the most defensible open direction along Axis 3 for a NeurIPS-level contribution. We expand on this in §8.

---

## 5. Axis 4: Sparsity, Merging, and Mixture-of-LoRAs

Axis 4 addresses a composition problem rather than a representation problem: given multiple LoRA adapters — whether for multiple tasks, multiple experts, or sparse fine-grained updates — how should they interact?

### 5.1 Sparsity

SHiRA (ICML'24, 2406.13175) makes a revealing argument: rather than using low-rank matrices, it applies a sparse $1\text{–}2\%$ mask directly to full-rank weights. The method is competitive with LoRA at matched parameter counts, which suggests that low-rank structure is not uniquely necessary — sparsity in the full-rank representation can substitute. LoRI (COLM'25, 2504.07448) combines frozen random $A$ with sparse $B$ under the constraint that different tasks use disjoint supports; this design choice turns out to solve the multi-task merging problem essentially by construction. SparseLoRA (ICML'25, 2506.16500) explores *computational* sparsity rather than parametric: the full $(A, B)$ are retained, but only a top-$k$ activation subset is used at each token, reducing inference cost without reducing capacity.

### 5.2 Merging and Task Arithmetic

A core observation of Ilharco et al. (2212.04089) is that fine-tuning produces a "task vector" $\tau_i = \theta_i - \theta_0$, and that simple arithmetic over task vectors — $\theta = \theta_0 + \sum_i \lambda_i \tau_i$ — composes capabilities from multiple fine-tunes. Naive task arithmetic suffers from destructive interference when task vectors overlap in weight space, which has motivated a series of methods to manage this interference.

TIES (NeurIPS'23, 2306.01708) introduces a three-step protocol: **Trim** small-magnitude entries, **Elect** the sign at conflicting positions by majority, and **Disjoint merge** at non-conflicting positions. DARE (ICML'24, 2311.03099) takes a different approach: it randomly drops entries of each $\tau_i$ with probability $p$ and rescales by $1/(1-p)$ to preserve the expectation. The dropout-style sparsification reduces interference without sign-conflict logic. LoraHub (COLM'24, 2307.13269) performs gradient-free CMA-ES search over mixing coefficients using a few-shot validation set.

Two recent NeurIPS'25 papers diagnose the geometric structure of the interference problem. Core Space (2505.18292) projects LoRA updates into a shared orthonormal core basis before merging, preserving information that is lost under naive averaging. Crowded-in-B-Space (2505.23873) identifies an asymmetry between the $A$ and $B$ factors: across fine-tunes, $B$ columns cluster while $A$ rows spread out. B-space decorrelation before merging yields modest but consistent improvements. Taken together, these papers signal that the merging axis has matured enough that improvements now require precise geometric diagnosis rather than generic heuristics.

### 5.3 Mixture-of-Experts with LoRA

The MoE × LoRA intersection is densely populated. LoRAMoE (2312.09979) introduces $N$ LoRA experts with softmax routing and a localized balance loss that splits experts into "world-knowledge" and "task-specific" roles. MoLE (NeurIPS'24, 2404.13628) freezes a pool of pretrained LoRAs and learns only layer-wise gates, yielding compositional adaptation without training new experts. X-LoRA (2402.07148) performs a two-pass forward: the first pass produces hidden states that determine a layer-and-adapter-level scaling tensor for the second pass. HydraLoRA (NeurIPS'24, 2404.19245) adopts an asymmetric design — a single shared $A$ with multiple task-specific $B_i$ — reflecting the observation that $A$ serves as a shared input projection while $B$ is the task-specific output. AdaMoLE, MixLoRA, and MoLA constitute further variations along these lines.

### 5.4 Head- and Circuit-Granular LoRA

A smaller but conceptually important sub-family attaches LoRA at the level of individual attention heads or circuits rather than entire weight matrices. HS-LoRA ($\sim$2410.18130) is a head-granular parameter-sharing method; despite the name, it does not use mechanistic-interpretability signals to select heads. Circuit-aware LoRA (ICML'25, $\sim$2502.18356) uses attribution patching to identify task-relevant circuits and attaches LoRA only to nodes on the circuit, with rank proportional to attribution score. This paper's mechanism-level contribution is *placement* for efficiency during *learning*; it does not address unlearning, does not use group-lasso regularization, and does not evaluate robustness under adversarial retraining. We will return to this distinction in §6.

### 5.5 Synthesis

Axis 4 exhibits a pattern of rapid saturation followed by geometric refinement. Task arithmetic, TIES, DARE, and LoraHub established the problem; Core Space and Crowded-in-B-Space in 2025 represent the second generation, offering principled geometric accounts of why naive composition fails. The MoE-LoRA sub-axis is similarly crowded, with most recent work providing incremental improvements to routing, expert specialization, or parameter sharing. Head-granular and circuit-aware LoRA remain relatively sparse, and — as we show below — the combination of pathway-structured LoRA with unlearning objectives constitutes a genuine open problem.

---

## 6. Axis 5: Interpretability, Unlearning, and Safety

Axis 5 differs from the previous axes in that it concerns **what LoRA is used for** rather than how it is parameterized or optimized. This axis is considerably younger — most activity dates from 2024 onwards — and its boundaries are still being defined.

### 6.1 Mechanistic-Interpretability-Guided Adaptation

A small but growing literature uses mechanistic-interpretability tools — circuit discovery, attribution patching, sparse autoencoders — to guide where and how LoRA is applied. Circuit-aware LoRA uses attribution patching for placement. Causal Head Gating learns continuous head gates via interchange interventions. Sufficient Subcircuits uses ACDC to identify minimal circuits supporting a task before adapting them. SAE-guided adapters constrain LoRA updates to span or suppress specific sparse-autoencoder feature directions. Representation Fine-Tuning (ReFT; NeurIPS'24, 2404.03592) takes a different stance — it modifies representations at specific $(\text{layer}, \text{token})$ positions rather than weights — but shares the interpretability motivation.

Across these methods, the core idea is that fine-tuning does not need to act uniformly across the model. Which specific interpretability signal to use, and how to integrate it with standard PEFT objectives, remain relatively open.

### 6.2 LoRA for Unlearning

Unlearning — removing learned knowledge from a model — has become a major application area for LoRA-like methods, driven by the emergence of benchmarks such as TOFU (2401.06121), WMDP (2403.03218), and MUSE.

The first generation of unlearning methods falls into three families. Gradient ascent on the forget set is the simplest but collapses utility. Gradient Difference and Negative Preference Optimization (NPO) balance forget and retain objectives but remain unstable. Representation Misdirection for Unlearning (RMU; WMDP paper, 2403.03218) perturbs internal activations on forget data toward random directions at chosen layers, typically implemented as a LoRA on those layers' MLPs. Saliency-based Unlearning (SalUn; ICLR'24 Spotlight, 2310.12508) computes gradient saliency on the forget loss and restricts updates to top-$|\text{grad}|$ weights, frequently combined with LoRA. Influence-function-based approaches (LIBU, 2411.x) select LoRA update directions that approximate leave-forget-out retraining via influence functions.

A critical development in 2024–2025 is the emergence of **robustness-of-unlearning** as the primary evaluation criterion. Lynch et al. (2402.16835) propose eight distinct evaluation methods for robust unlearning, and their key finding — replicated across WMDP, TOFU, and MUSE — is that most LoRA-based unlearning methods collapse under 10–100 steps of benign relearning, recovering 80–95% of forgotten capability. Deeb and Roger (2407.01920) formalize this as the question "Do unlearning methods remove information?" and answer negatively for most existing methods. The relearning attack from ICLR'25 ("Jogging the Memory") has become a standard evaluation.

Taken together, this literature identifies a concrete open problem: no existing LoRA-based unlearning method combines (i) **mechanistically structured forget targets** — e.g., group-sparse pathway ablation rather than unstructured weight masking — with (ii) **training-time robustness** to relearning attacks, e.g., through latent adversarial training. HS-LoRA does not do unlearning; Circuit-aware LoRA does not use group-lasso or address unlearning. We elaborate in §8.

### 6.3 Safety and Refusal

A distinct safety-adjacent literature includes Circuit Breakers (2406.04313), which uses LoRA on refusal-relevant layers to reroute representations away from harmful generations and achieves strong robustness against GCG and PAIR attacks; SafeLoRA (NeurIPS'24, 2405.16833), which projects fine-tuned LoRA deltas onto a "safety subspace" derived from the delta between aligned and unaligned models; and Lisa/Vaccine, which use perturbation-aware training to resist harmful fine-tuning.

### 6.4 Knowledge Editing

ROME (Meng et al. 2022) and MEMIT (2023) are rank-1 direct edits at specific layers; they are not LoRA but are mathematically adjacent and often serve as baselines. MELO (AAAI'24) and LoRAEdit apply LoRA-style updates with locality constraints for knowledge editing, evaluated on the KnowEdit benchmark (metrics: efficacy, generalization, locality, portability).

---

## 7. Axis 6: Theory, Quantization, and Federated Learning

### 7.1 Theoretical Foundations

The theoretical literature on LoRA is smaller than the empirical literature but has clarified several fundamental questions.

**Expressiveness.** Zeng and Lee (ICLR'24, 2310.17513) prove that for feed-forward networks, LoRA with $r \geq (\text{width} \times \text{depth-ratio})$ can exactly represent any target network of smaller depth, establishing a rank-width trade-off. The corresponding result for softmax-attention architectures remains open — a non-trivial gap given that LoRA is used almost exclusively on Transformers.

**Optimization landscape.** Jang et al. (ICML'24, 2402.11867) prove that in the NTK regime, LoRA with $r \geq \sqrt{N}$ has no spurious local minima, and provide a generalization bound $O(\sqrt{r/N})$. Whether this extends beyond the lazy regime to feature-learning is not known.

**Asymmetry.** Several papers identify a fundamental asymmetry between $A$ and $B$. Zhu et al. (2402.16842) argue that $B$ matters more than $A$ for generalization. Hayou et al. (2402.12354), from independent infinite-width scaling arguments, derive the specific prescription $\eta_B = 16 \eta_A$. This is one of the cleanest theory-to-method pipelines in the LoRA literature.

**Intruder dimensions.** Shuttleworth et al. (2410.21228) compare the SVDs of LoRA-adapted and fully-fine-tuned weight matrices and identify the emergence of high-ranking singular vectors in LoRA that are absent in full fine-tuning. These "intruder dimensions" are causally implicated in forgetting of pretraining knowledge and OOD degradation. Crucially, the paper also demonstrates that post-hoc scaling of intruder dimensions substantially mitigates the issue, partially closing what might otherwise appear to be a pure open problem. The remaining space concerns *training-time* prevention of intruder dimensions, which is narrower but non-trivial.

**Intrinsic dimensionality.** Aghajanyan et al. (2012.13255) empirically established — in the work that originally motivated LoRA — that fine-tuning of RoBERTa occupies a subspace of dimension $d_{90} \approx 200$.

### 7.2 Quantization

The quantization-LoRA combination has matured considerably since QLoRA's introduction. QLoRA (NeurIPS'23, 2305.14314) combines 4-bit NF4 quantization with paged optimizers and double quantization to enable 65B-parameter fine-tuning on a single 48GB GPU. LoftQ (ICLR'24, 2310.08659) addresses the initialization mismatch in quantized training by alternating between quantization of $W - BA$ and SVD of $W - Q$, yielding $1\text{–}8$ Rouge improvements at 2–4 bits. LQ-LoRA (ICLR'24, 2311.12023) uses Fisher-weighted mixed-precision decomposition via integer linear programming. QA-LoRA (ICLR'24, 2309.14717) designs the quantization granularity so that LoRA can be merged into a quantized backbone without dequantization, enabling true INT4 inference with 2–3× speedup. IR-QLoRA, ApiQ, and LR-QAT extend these ideas to 2-bit and sub-2-bit regimes.

### 7.3 Federated Fine-Tuning

The federated LoRA literature has converged on a specific technical issue: aggregating client-side LoRA updates via FedAvg is biased, because $\text{Avg}(B_i A_i) \neq \text{Avg}(B_i) \cdot \text{Avg}(A_i)$. FedIT (2305.05644) acknowledges but does not solve this. FLoRA (NeurIPS'24, 2409.05976) sidesteps the issue by stacking client factors as block matrices rather than averaging, which also accommodates heterogeneous ranks across clients, at the cost of communication growing linearly in the number of clients. LoRA-FAIR (ICCV'25, 2411.14961) restores the FedAvg communication budget while correcting aggregation and initialization drift through a residual correction term. HetLoRA addresses heterogeneous-rank clients via zero-padding and rank-pruning.

### 7.4 Continual and Multi-Task

O-LoRA (EMNLP'23, 2310.14152) imposes orthogonality constraints between task-specific LoRA subspaces to resist catastrophic forgetting, with empirical capacity saturating around 15 tasks. InfLoRA (CVPR'24, 2404.00228) projects gradients into the subspace orthogonal to past task input spans. I-LoRA (2408.14961) maintains fast and slow LoRA copies via EMA in a dual-memory design.

---

## 8. Analysis of Open Problems

We now synthesize the axis-level observations into a map of open problems, ranked by the combination of (i) the degree to which the mechanism is genuinely unoccupied and (ii) the likely magnitude of a self-contained contribution along that direction.

### 8.1 Tier 1: Mechanistically Distinct Open Problems

**Open Problem 1 (OP1): Natural gradient in the LoRA rank-$r$ subspace.**

As discussed in §4.3, the use of the loss Fisher $F = \mathbb{E}[\nabla L \nabla L^\top]$ as a training-time preconditioner in the LoRA parameter vector space is essentially unoccupied. This is not a topic gap but a mechanism gap: every method in the nearest neighborhood uses some structurally different object (parameterization metric, single-gradient match, post-hoc posterior, diagonal curvature, or NTK). The computational barrier that historically prevented natural gradient on full models — $O(d^2)$ Fisher storage — evaporates in the LoRA subspace, where Woodbury reduces the inverse to a small $m \times m$ solve.

A minimum-viable contribution along this direction would demonstrate: (a) wall-clock speedup over AdamW, LoRA-Pro, and Riemannian Preconditioned LoRA on GLUE, GSM8K, and MMLU; (b) a convergence rate derivation following the Amari natural-gradient framework; (c) compatibility with existing quantization (QLoRA / LoftQ) and initialization (LoRA-GA, PiSSA) pipelines. The principal risk is classification as "yet another preconditioner"; the defense is a clean mechanism-level argument for why Fisher in the rank-$r$ subspace is qualitatively different from the existing five classes.

**Open Problem 2 (OP2): Mechanistically structured, relearning-robust unlearning.**

As discussed in §6.2, no existing method combines (i) pathway-structured LoRA updates at the $(\text{layer}, \text{head}, \text{projection type})$ level with group-lasso regularization on pathway norms, (ii) a forget objective that operates at the level of these structured pathways, and (iii) training-time robustness to relearning attacks via latent adversarial training. The nearest neighbors — SalUn (weight-level masking), RMU (activation perturbation), Circuit-aware LoRA (placement for learning, not unlearning), HS-LoRA (parameter sharing without attribution), LIBU (influence functions without structure) — each address a strict subset of this triple.

A minimum-viable contribution would demonstrate pathway-level forget effectiveness on WMDP and TOFU comparable to RMU, but with post-relearning recovery rates substantially below the current 80–95% baseline established by Lynch et al. (2402.16835). The principal risk is that the method is empirically strong but lacks a clean theoretical hook; this is mitigable by grounding the pathway structure in a sparsity-recovery argument and the adversarial training in the standard LAT framework.

**Open Problem 3 (OP3): Mechanistic-interpretability signals for rank allocation.**

As discussed in §3.2, existing rank allocation methods use sensitivity scores (AdaLoRA), gate thresholds (SoRA), adapter output norms (LoRA-Drop), or SHAP-like ablation (ALoRA). None uses the causal attribution signals developed in the mechanistic-interpretability literature — path patching, attribution patching, integrated gradients on circuit edges. The difference matters mechanistically: attribution patching measures the causal contribution of a component's activation to a task loss, while sensitivity measures the derivative of the loss with respect to parameter values. These are conceptually orthogonal signals that should induce different allocation policies.

This contribution is likely narrower than OP1 or OP2 and may be most productive as a sub-component of OP2 (where pathway selection and unlearning are jointly optimized) rather than as a standalone paper.

### 8.2 Tier 2: Secondary Open Problems

**OP4: Gradient covariance initialization.** LoRA-GA uses $\mathbb{E}[\nabla W]$; no method uses $\text{Cov}(\nabla W)$ for initialization, though the covariance captures stochastic structure that the mean discards. Naturally pairs with OP1 — covariance initialization is the correct dual of Fisher-preconditioned training.

**OP5: Initialization-time gauge fixing.** LoRA-RITE addresses the $\text{GL}(r)$ gauge redundancy in the optimizer; eliminating this redundancy at initialization is simpler and unexplored. Most natural as a secondary contribution within a paper whose primary contribution is elsewhere.

### 8.3 Tier 3: Risky or Partially Occupied Directions

**OP6: Learnable atom dictionary with per-pathway group-lasso selector.** VB-LoRA (NeurIPS'24, §2.1) occupies the core of this direction: shared global vector bank with sparse top-$k$ admixture selection. A variant using group-lasso by projection type (Q/K/V/O/FFN) rather than top-$k$ softmax, and rank-1 atom granularity rather than sub-vector granularity, is differentiable in principle, but the reviewer response will hinge on whether the distinction is substantive or incremental. We do not recommend this direction without a sharply articulated differentiation.

**OP7: Training-time intruder-dimension prevention.** Shuttleworth et al. (2410.21228) already provide a post-hoc scaling mitigation. Training-time regularization that prevents intruder dimensions from forming in the first place is a narrower but non-trivial question.

**OP8: Advanced LoRA × advanced quantization.** Combinatorial space of DoRA + 2-bit LoftQ / ApiQ, VeRA with quantized shared matrices, rotation-based quantization (QuIP#, SpinQuant) composed with LoRA-FA or LoRA+. Generally incremental.

### 8.4 Saturated Regions

The following regions have reached a maturity where further contributions require either a specific deployment niche or a fundamentally new theoretical angle:

- Riemannian / Stiefel manifold methods on LoRA (StelLA NeurIPS'25 Spotlight, Stiefel-LoRA, Riemannian Preconditioned LoRA, Muon-LoRA).
- Tensor decompositions of LoRA (LoTR, LoRTA, TensLoRA, TensorLLM, SuperLoRA, block-Tucker).
- Shared-basis and bank-based LoRA (VeRA, VB-LoRA, HydraLoRA, Tied-LoRA).
- Gradient- and SVD-based initialization (LoRA-GA, LoRA-Pro, LoRA-One, PiSSA, MiLoRA, CorDA, EVA).
- Mixture-of-LoRA routing (LoRAMoE, MoLE, X-LoRA, HydraLoRA, MixLoRA, MoLA, AdaMoLE).
- LoRA merging and task arithmetic (TIES, DARE, Core Space, Crowded-in-B-Space).

Researchers entering these regions should expect a narrow, specific niche to be the only path to publication.

---

## 9. Recommended Research Directions

Based on the above analysis, we identify two primary directions that combine mechanistic distinctness, theoretical tractability, and experimental feasibility:

**Direction A: Second-Order LoRA.** A paper combining OP1 (natural gradient in the LoRA subspace) with OP4 (gradient-covariance initialization) and potentially OP5 (initialization-time gauge fixing). The unifying intellectual claim is that *second-order information — initialization from gradient covariance, training-time preconditioning by the empirical Fisher — becomes computationally tractable inside the LoRA subspace in a way it is not at the full-model scale*. Theoretical contributions are within reach (Amari-style convergence, PAC-Bayes bounds over low-rank posteriors). Experimental baselines are well-defined (AdamW, LoRA-Pro, Riemannian Preconditioned LoRA, Sophia, StelLA).

**Direction B: Circuit-Aligned Surgical Unlearning.** A paper combining OP2 (mechanistically structured, relearning-robust unlearning) with OP3 (attribution-based rank allocation). The intellectual claim is that *unlearning should operate at the level of interpretable circuit pathways, and that robustness to relearning requires adversarial training at training time, not just post-hoc evaluation*. Evaluation uses WMDP, TOFU, MUSE under the Lynch et al. robustness battery. Theoretical hooks are less clean than Direction A but exist (sparsity-recovery under group-lasso, LAT robustness).

These two directions are largely independent and could, with sufficient resources, be pursued in parallel.

**Assessment.** Direction A is the stronger standalone contribution: its mechanism-level differentiation from all neighbors is clean, its theoretical framework is well-established, and its experimental protocol is uncontroversial. Direction B is more timely given the growth of safety and unlearning benchmarks but carries higher evaluation-protocol risk. We recommend Direction A as the primary path.

---

## Appendix: Index of Cited Papers by arXiv ID

Papers are listed by arXiv identifier for ease of reference. Axis indicates the primary locus of contribution.

| arXiv ID | Short name | Axis |
|---|---|---|
| 2012.13255 | Intrinsic dimensionality (Aghajanyan) | 7.1 |
| 2106.09685 | LoRA | baseline |
| 2108.06098 | LoHa / FedPara | 2.2 |
| 2210.04284 | SparseAdapter | 5.1 |
| 2210.07558 | DyLoRA | 3.2 |
| 2212.04089 | Task Arithmetic | 5.2 |
| 2303.10512 | AdaLoRA | 3.2 |
| 2305.14314 | QLoRA | 7.2 |
| 2306.01708 | TIES-Merging | 5.2 |
| 2306.07280 | OFT | 2.2 |
| 2307.05695 | ReLoRA | 2.1 |
| 2307.13269 | LoraHub | 5.2 |
| 2308.03303 | LoRA-FA | 2.1 |
| 2308.06522 | SLoRA (federated) | 7.3 |
| 2308.12043 | IncreLoRA | 3.2 |
| 2308.13111 | Laplace-LoRA | 4 |
| 2309.02411 | Delta-LoRA | 3.1 |
| 2309.14717 | QA-LoRA | 7.2 |
| 2309.14859 | LoKr | 2.2 |
| 2310.08659 | LoftQ | 7.2 |
| 2310.11454 | VeRA | 2.1 |
| 2310.12508 | SalUn | 6.2 |
| 2310.14152 | O-LoRA | 7.4 |
| 2310.17513 | Expressive Power of LoRA (Zeng & Lee) | 7.1 |
| 2311.03099 | DARE | 5.2 |
| 2311.06243 | BOFT | 2.2 |
| 2311.09578 | Tied-LoRA | 2.1 |
| 2311.11696 | SoRA | 3.2 / 5.1 |
| 2311.12023 | LQ-LoRA | 7.2 |
| 2402.01376 | LoTR | 2.3 |
| 2402.02347 | Riemannian Preconditioned LoRA | 4 |
| 2402.07148 | X-LoRA | 5.3 |
| 2402.07721 | LoRA-Drop | 3.2 |
| 2402.09353 | DoRA | 2.1 |
| 2402.11867 | NTK LoRA landscape (Jang et al.) | 7.1 |
| 2402.12354 | LoRA+ | 4 |
| 2402.16835 | Lynch et al. (robust unlearning eval) | 6.2 |
| 2402.16842 | LoRA asymmetry (Zhu et al.) | 7.1 |
| 2402.17263 | MELoRA | 2.1 |
| 2403.03218 | WMDP + RMU | 6.2 |
| 2403.11887 | SuperLoRA | 2.3 |
| 2403.16187 | ALoRA | 3.2 |
| 2404.00228 | InfLoRA | 7.4 |
| 2404.02948 | PiSSA | 3.1 |
| 2404.03592 | ReFT | 6.1 |
| 2404.13628 | MoLE | 5.3 |
| 2404.15159 | MixLoRA | 5.3 |
| 2404.19245 | HydraLoRA | 5.3 |
| 2405.03003 | FourierFT | 2.2 |
| 2405.13053 | LoRA Soups | 5.2 |
| 2405.15179 | **VB-LoRA** | 2.1 |
| 2405.16833 | SafeLoRA | 6.3 |
| 2405.17484 | HRA | 2.2 |
| 2405.17604 | LoRA-XS | 2.1 |
| 2405.19597 | SVFT | 2.2 |
| 2406.01775 | OLoRA | 3.1 |
| 2406.04313 | Circuit Breakers | 6.3 |
| 2406.05223 | CorDA | 3.1 |
| 2406.09044 | MiLoRA | 3.1 |
| 2406.11909 | MoSLoRA / MiSS | 2.1 |
| 2406.13175 | SHiRA | 5.1 |
| 2407.01920 | Deeb & Roger (info-removal) | 6.2 |
| 2407.05000 | LoRA-GA | 3.1 |
| 2407.18242 | LoRA-Pro | 4 |
| 2408.10280 | NoRA | 2.1 |
| 2408.14961 | I-LoRA | 7.4 |
| 2409.05976 | FLoRA (federated) | 7.3 |
| 2410.04060 | LoRTA | 2.3 |
| 2410.07170 | EVA | 3.1 |
| 2410.20625 | LoRA-RITE | 3.2 |
| 2410.21228 | Illusion of Equivalence (Shuttleworth) | 7.1 |
| 2411.14961 | LoRA-FAIR (federated) | 7.3 |
| 2412.06071 | KaSA | 3.1 |
| 2502.01235 | LoRA-One | 3.1 |
| ~2502.18356 | Circuit-aware LoRA | 5.4 |
| 2504.07448 | LoRI | 5.1 |
| 2505.12455 | AltLoRA | 4 |
| 2505.18292 | Core Space (merging) | 5.2 |
| 2505.23873 | Crowded-in-B-Space | 5.2 |
| 2506.16500 | SparseLoRA | 5.1 |
| 2506.16787 | SeLoRA | 3.3 |
| 2506.18267 | ARD-LoRA | 3.2 |
| 2508.17901 | Stiefel-LoRA | 4 |
| 2510.00192 | PrunedLoRA | 3.3 |
| 2510.24561 | LoRA-DA | 3.1 |

**Note on arXiv identifiers.** Several identifiers for 2025–2026 papers are approximate to within a few digits, reflecting the state of the literature at the time of writing. Verification against the arXiv listing is recommended before citation.
