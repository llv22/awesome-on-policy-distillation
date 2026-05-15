# ⚗️ Awesome On-Policy Distillation

<p align="center">
  <a href="https://awesome.re"><img src="https://img.shields.io/badge/Awesome-%E2%9A%97%EF%B8%8F_On--Policy_Distillation-000000?style=for-the-badge&labelColor=000000" alt="Awesome On-Policy Distillation"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/stargazers"><img src="https://img.shields.io/github/stars/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Stars&labelColor=000000&color=000000" alt="GitHub Stars"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/network/members"><img src="https://img.shields.io/github/forks/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Forks&labelColor=000000&color=000000" alt="GitHub Forks"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/commits"><img src="https://img.shields.io/github/last-commit/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Last%20Commit&labelColor=000000&color=000000" alt="Last Commit"></a>
</p>

A curated collection of papers, technical reports, frameworks, and tools for on-policy distillation (OPD) of large language models.

> **On-policy distillation** trains a student on samples from its own evolving policy, while a teacher (external, privileged, or self-conditioned) provides dense supervision on those same samples.

OPD sits between supervised fine-tuning and reinforcement learning. Unlike off-policy KD, the student trains on its *own* generations, closing the train-inference distribution gap. Unlike RL, the student receives dense token-level teacher guidance rather than sparse rewards.

As of 2026, OPD is a standard post-training primitive at Alibaba (Qwen3), Xiaomi (MiMo), Zhipu (GLM-5), NVIDIA (Nemotron-Cascade 2), and others.

## Contents

- [Quick Start by Role](#quick-start-by-role)
- [Start Here](#start-here)
- [Taxonomy](#taxonomy)
  - [By Teacher Type](#by-teacher-type)
  - [By Primary Goal](#by-primary-goal)
- [Surveys](#surveys)
- [Core OPD Papers](#core-opd-papers)
  - [Foundations](#foundations)
  - [Gap-Bridging](#gap-bridging)
  - [Stability and Objective Design](#stability-and-objective-design)
  - [Self-Distillation](#self-distillation)
  - [Context and Experience Internalization](#context-and-experience-internalization)
  - [Efficiency Variants](#efficiency-variants)
- [Adjacent and Enabling Work](#adjacent-and-enabling-work)
  - [Cross-Tokenizer and Model-Family Enablers](#cross-tokenizer-and-model-family-enablers)
  - [Mismatch Mitigation and Student Quality](#mismatch-mitigation-and-student-quality)
  - [Preference, Reward-Guided, and Hybrid RL+KD](#preference-reward-guided-and-hybrid-rlkd)
  - [Self-Play and Iterative Bootstrapping](#self-play-and-iterative-bootstrapping)
  - [Agent Distillation, Multimodal, and Other Extensions](#agent-distillation-multimodal-and-other-extensions)
  - [Speculative Decoding (Draft-Model Training)](#speculative-decoding-draft-model-training)
  - [Precursors](#precursors)
- [Technical Reports and Industrial Recipes](#technical-reports-and-industrial-recipes)
- [Frameworks, Tools, and Implementations](#frameworks-tools-and-implementations)
  - [Training Frameworks](#training-frameworks)
  - [Implementations](#implementations)
  - [Essays, Blog Posts, and Walkthroughs](#essays-blog-posts-and-walkthroughs)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Citation](#citation)

## Quick Start by Role

| Role | Start With | Key Resources |
|---|---|---|
| New to distillation | Definition above, then the [reading path](#start-here) | [Start Here](#start-here) |
| Researcher surveying the field | [Core OPD Papers](#core-opd-papers) for the canonical 21 | [Taxonomy](#taxonomy) for a structured map |
| Building an OPD pipeline | [TRL](https://huggingface.co/docs/trl)'s GKD trainer to start | [NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html) or [veRL](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) for scale |
| Evaluating OPD for post-training | [Technical Reports](#technical-reports-and-industrial-recipes) for who is shipping it | [Core OPD Papers](#core-opd-papers) for algorithmic foundations |

> **Key decision:** Do you have access to teacher logits? If yes, start with white-box methods (GKD, Veto, Entropy-Aware OPD). If no, see black-box methods (GAD, OVD) or self-distillation (OPSD, SDFT).

## Start Here

The fastest path to understanding the field:

0. **Survey** — [OPD Survey](https://arxiv.org/abs/2604.00626).
   Comprehensive map of the field: taxonomy, methods, and open problems.
1. **Foundations** — [MiniLLM](https://arxiv.org/abs/2306.08543) and [GKD](https://arxiv.org/abs/2306.13649).
   You will understand the basic student-rollout + teacher-supervision loop.
2. **Practical intuition** — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/).
   The clearest end-to-end explanation of why and when to use OPD.
3. **Limitations of vanilla OPD** — [Black-Box OPD](https://arxiv.org/abs/2511.10643), [Veto](https://arxiv.org/abs/2601.07155), [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079), [Revisiting OPD](https://arxiv.org/abs/2603.25562).
   You will learn what breaks: instability, diversity collapse, no-logit settings, sampled-token failure modes.
4. **Self-distillation** — [OPSD](https://arxiv.org/abs/2601.18734), [SDFT](https://arxiv.org/abs/2601.19897), [SDPO](https://arxiv.org/abs/2601.20802).
   Drop the external teacher entirely.
5. **Context and experience** — [OPCD](https://arxiv.org/abs/2602.12275) and [OEL](https://arxiv.org/abs/2603.16856).
   Distill prompts, traces, and deployment experience into weights.
6. **2026 efficiency frontier** — [Prefix OPD](https://arxiv.org/abs/2602.15260), [OVD](https://arxiv.org/abs/2601.21968), [PACED](https://arxiv.org/abs/2603.11178), [REOPOLD](https://arxiv.org/abs/2603.11137).
   Cut compute 2x-47x.
7. **Industrial patterns** — [Qwen3](https://arxiv.org/abs/2505.09388), [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), [GLM-5](https://arxiv.org/abs/2602.15763).
   How labs deploy OPD in production.

## Taxonomy

### By Teacher Type

| Teacher Type | Papers |
|---|---|
| External white-box | MiniLLM, GKD, Veto, Entropy-Aware OPD, ExOPD, REOPOLD, PACED, Prefix OPD, Revisiting OPD, Rethinking OPD, Lightning OPD, Uni-OPD, SOD, AOPD, vOPD, NPD, Prune-OPD, EffOPD, CoDistill-GRPO, Rock Tokens |
| External black-box | Black-Box OPD / GAD, OVD, ROPD |
| Self-teacher with privileged context | OPSD, SDFT, SDPO, OPSDC, GATES, pi-Distill, RLSD, SDZero, OGLS-SD, PBSD, UniSD, ATESD, RLRT |
| Context-conditioned | OPCD, OEL |
| Multiple / lifecycle teachers | MiMo-V2-Flash MOPD, GLM-5, Qwen3, Baichuan-M3, DeepSeek-V4, CoPD, MAD-OPD, KAT-Coder-V2 |

### By Primary Goal

| Goal | Papers |
|---|---|
| Compression / strong-to-weak transfer | MiniLLM, GKD, Qwen3, Prefix OPD, Rethinking OPD, Lightning OPD |
| Post-RL consolidation / skill integration | MiMo MOPD, GLM-5, ExOPD, CoPD |
| Continual learning | SDFT, OPCD, OEL |
| RL replacement / augmentation | SDPO, RLTF-SD, RLAD, REOPOLD, RLSD, SDZero, OGLS-SD, PBSD, CoDistill-GRPO, RLRT |
| Reasoning compression | OPSDC |
| Black-box distillation | GAD, OVD, ROPD |

Many papers span multiple categories. The taxonomy is for orientation, not strict partitioning.

## Surveys

- **A Survey of On-Policy Distillation for Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2604.00626-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.00626)
  - *(2026)* First dedicated OPD survey; organizes methods by feedback signal, teacher access mode, and loss scope.

## Core OPD Papers

The ~21 papers that define on-policy distillation for LLMs.

### Foundations

- **MiniLLM: On-Policy Distillation of Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2306.08543-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.08543)
  - *(2023)* Reverse-KL framing for generative LMs; the paper that named the field.
- **GKD: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** [![arXiv](https://img.shields.io/badge/arXiv-2306.13649-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.13649)
  - *(2023)* Unifying formulation spanning on-/off-policy mixtures with flexible divergences.

### Gap-Bridging

- **Speculative Knowledge Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2410.11325-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.11325)
  - *(2024)* Interleaved teacher/student sampling to mitigate poor student rollout quality.
- **Black-Box On-Policy Distillation of Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2511.10643-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.10643)
  - *(2025)* GAD: black-box OPD via discriminator-based reward on student rollouts; no teacher logits needed.
- **SOD: Step-wise On-policy Distillation for Small Language Model Agents** [![arXiv](https://img.shields.io/badge/arXiv-2605.07725-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.07725)
  - *(2026)* Step-wise OPD for tool-integrated SLM agents that reweights teacher guidance by step-level student-teacher divergence to avoid tool-induced cascade drift.
- **MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate** [![arXiv](https://img.shields.io/badge/arXiv-2605.01347-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.01347)
  - *(2026)* Recasts the OPD teacher as a deliberative collective whose post-debate consensus supplies token-level supervision on student rollouts; extends to agentic tasks via OPAD with step-level sampling and a task-adaptive divergence principle.
- **ROPD: Rubric-based On-policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2605.07396-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.07396)
  - *(2026)* Black-box OPD that induces prompt-specific semantic rubrics from teacher-student contrasts and uses them to score student rollouts, replacing teacher logits with rubric-based reward.

### Stability and Objective Design

- **Veto: Stable On-Policy Distillation through Adaptive Target Reformulation** [![arXiv](https://img.shields.io/badge/arXiv-2601.07155-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.07155)
  - *(2026)* Intermediate target distribution in logit space to stabilize training.
- **Entropy-Aware On-Policy Distillation of Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2603.07079-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.07079)
  - *(2026)* Forward-KL on high-entropy teacher tokens to preserve output diversity.
- **ExOPD: Learning beyond Teacher via Generalized On-Policy Distillation with Reward Extrapolation** [![arXiv](https://img.shields.io/badge/arXiv-2602.12125-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.12125)
  - *(2026)* Casts OPD as dense KL-constrained RL; reward scaling enables teacher-surpassing behavior.
- **REOPOLD: Scaling Reasoning Efficiently via Relaxed On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2603.11137-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.11137)
  - *(2026)* Relaxes strict imitation with reward clipping, entropy-based dynamic sampling, and explore-to-refine training.
- **PACED: Distillation at the Frontier of Student Competence** [![arXiv](https://img.shields.io/badge/arXiv-2603.11178-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.11178)
  - *(2026)* Pass-rate weighting focuses learning on the student's competence frontier.
- **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes** [![arXiv](https://img.shields.io/badge/arXiv-2603.25562-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.25562)
  - *(2026)* Truncated reverse-KL with teacher top-K local support matching; fixes imbalanced signals, unreliable teacher guidance, and tokenizer mismatch in sampled-token OPD.
- **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe** [![arXiv](https://img.shields.io/badge/arXiv-2604.13016-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.13016)
  - *(2026)* Mechanistic analysis of OPD dynamics; identifies compatible thinking patterns and novel teacher capability as success conditions; proposes off-policy cold start and teacher-aligned prompt selection for recovery.
- **The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2604.16830-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.16830)
  - *(2026)* Theoretical analysis of OPD-induced overconfidence (information asymmetry, entropy collapse, selection bias); CaOPD replaces confidence targets with student-grounded empirical success rates to decouple capability from calibration.
- **Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2604.08527-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.08527)
  - *(2026)* Diagnoses repetition-driven length inflation in iterative OPD; Stable-OPD adds divergence constraints and a rollout-mixture anchor with golden data.
- **Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe** [![arXiv](https://img.shields.io/badge/arXiv-2605.03677-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.03677)
  - *(2026)* Identifies insufficient state exploration and unreliable teacher supervision as OPD bottlenecks; pairs offline difficulty-aware plus online correctness-aware data balancing with an outcome-guided margin calibration that keeps token-level teacher scores order-consistent with outcome reward across LLM and MLLM settings.
- **AOPD: Asymmetric On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2605.06387-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.06387)
  - *(2026)* Diagnoses high variance, vanishing gradients, and exploration bottlenecks in standard OPD; replaces ineffective negative reinforcement with localized teacher-distribution matching in non-positive advantage regions while preserving positive reinforcement.
- **vOPD: On-Policy Distillation with a Control Variate Baseline** [![arXiv](https://img.shields.io/badge/arXiv-2605.07865-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.07865)
  - *(2026)* Stabilizes single-sample OPD by subtracting a closed-form per-token reverse-KL value baseline derived from the already-computed forward pass; unbiased lower-variance estimator with no extra critic.
- **Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why** [![arXiv](https://img.shields.io/badge/arXiv-2605.10889-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10889)
  - *(2026)* Training-free per-token, per-question, per-teacher diagnostic computing gradient alignment (cosine of distillation gradient with an ideal success-maximizing gradient derived from empirical pass rates via targeted rollouts); finds distillation guidance is more reliable on incorrect rollouts than correct ones, the best teacher/context depends jointly on student capacity and task (no universal recipe), and student-teacher divergence is a weak but cheap necessary-condition filter for alignment.
- **The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes** [![arXiv](https://img.shields.io/badge/arXiv-2605.11182-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.11182)
  - *(2026)* Empirical study of when OPD and OPSD work and fail across reasoning, system-prompt internalization, and alignment; identifies three failure mechanisms (student-prefix-induced teacher-state mismatch, biased Top-K reverse-KL gradients, and OPSD's PI-free aggregation across instance-specific privileged information) and three stabilizers (stop-gradient Top-K KL, RLVR-adapted teachers, and SFT-stabilized students for length and format).
- **Rock Tokens: Cornerstones or Stumbling Blocks? Deciphering the High-Loss Tokens in On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2605.09253-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.09253) [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/YuxuanJiang1/Rock-Token)
  - *(2026)* Identifies "Rock Tokens" — high-loss tokens that persist (up to 18% of outputs) even after OPD apparent convergence, dominating gradient norms yet remaining stagnant under teacher correction; causal interventions show they contribute negligibly to reasoning, so strategically masking them in the per-token KL objective streamlines alignment and challenges uniform token weighting.

### Self-Distillation

- **OPSD: Self-Distilled Reasoner** [![arXiv](https://img.shields.io/badge/arXiv-2601.18734-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.18734)
  - *(2026)* Single model as both teacher and student via privileged information; no external teacher required.
- **SDFT: Self-Distillation Enables Continual Learning** [![arXiv](https://img.shields.io/badge/arXiv-2601.19897-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.19897)
  - *(2026)* Demonstration-conditioned self-teaching for continual learning with less forgetting.
- **SDPO: Reinforcement Learning via Self-Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2601.20802-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.20802)
  - *(2026)* Converts textual feedback into dense self-teacher signals for RL-like training.
- **Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?** [![arXiv](https://img.shields.io/badge/arXiv-2603.24472-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.24472)
  - *(2026)* Traces self-distillation failures to suppression of epistemic verbalization; task coverage determines whether conciseness helps or hurts.
- **OPSDC: On-Policy Self-Distillation for Reasoning Compression** [![arXiv](https://img.shields.io/badge/arXiv-2603.05433-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.05433)
  - *(2026)* Compresses verbose reasoning using concise privileged self-teachers.
- **GATES: Self-Distillation under Privileged Context with Consensus Gating** [![arXiv](https://img.shields.io/badge/arXiv-2602.20574-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.20574)
  - *(2026)* Consensus-gated asymmetric-context self-distillation without labels or rewards.
- **HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2603.23871-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23871)
  - *(2026)* Privileged self-distillation targeting cliff prompts where RL gradients vanish; provably recovers the KL-regularized optimal policy.
- **RLSD: Self-Distilled RLVR** [![arXiv](https://img.shields.io/badge/arXiv-2604.03128-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.03128)
  - *(2026)* Repurposes self-distillation as token-level credit assignment within GRPO; proves OPSD-style distribution matching under information asymmetry induces irreducible privileged information leakage.
- **SDZero: Self-Revision Turns Binary Rewards into Dense Supervision** [![arXiv](https://img.shields.io/badge/arXiv-2604.12002-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.12002)
  - *(2026)* Generator-reviser dual-role self-distillation requiring only binary rewards; the reviser converts outcome-level feedback into token-level supervision on student rollouts without any external teacher or demonstrations.
- **OPSDL: On-Policy Self-Distillation for Long-Context Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2604.17535-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.17535)
  - *(2026)* Long-context self-distillation: short-context distribution of the same model serves as a co-evolving token-level reverse-KL teacher for student rollouts under long context.
- **PBSD: Preference-Based Self-Distillation — Beyond KL Matching via Reward Regularization** [![arXiv](https://img.shields.io/badge/arXiv-2605.05040-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.05040)
  - *(2026)* Replaces direct KL matching in on-policy self-distillation with a reward-regularized objective whose analytic optimum is a reward-tilted teacher distribution, instantiated as DPO-style preference learning between context-augmented teacher positives and on-policy student negatives; provably improves over the teacher and stabilizes training.
- **UniSD: Towards a Unified Self-Distillation Framework for Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2605.06597-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.06597) [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ahren09/UniSD)
  - *(2026)* Unifies autoregressive-LLM self-distillation along three axes — supervision reliability (multi-teacher agreement and token-level contrastive learning), representation alignment (feature matching), and training stability (EMA teacher and divergence clipping); the integrated UniSDfull pipeline improves base models by +5.4 and the strongest baseline by +2.8 across six benchmarks and three model families without any stronger external teacher.
- **OPSD Compresses What RLVR Teaches: A Post-RL Compaction Stage for Reasoning Models** [![arXiv](https://img.shields.io/badge/arXiv-2605.06188-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.06188)
  - *(2026)* Splits OPSD by rollout correctness on thinking-enabled math: correct-only training preserves accuracy and shortens responses, incorrect-only training damages accuracy, indicating the hindsight self-teacher reveals redundancy more reliably than missing reasoning steps; proposes SFT → RLVR → OPSD as a post-RL compaction stage.
- **ATESD: Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning** [![arXiv](https://img.shields.io/badge/arXiv-2605.11458-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.11458)
  - *(2026)* Identifies teacher-side exposure mismatch in OPSD — letting the privileged teacher see the full reference reasoning produces token targets too strong for the student to absorb — and treats the reveal ratio as a learnable control variable governed by a Beta-policy controller trained with a discounted learning-progress reward that handles OPD's delayed credit assignment; +0.95/+2.05/+2.33 Avg@12 over OPSD on AIME24, AIME25, HMMT25 for Qwen3-1.7B/4B/8B.
- **OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning** [![arXiv](https://img.shields.io/badge/arXiv-2605.12400-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.12400)
  - *(2026)* Diagnoses OPSD collapse as teacher-student mismatch from reflection-induced bias and privileged-context overconfidence; contrasts averaged teacher logits over correct vs. incorrect on-policy rollouts to form an outcome-guided steering direction added to the anchor teacher logits, calibrating token-level self-distillation supervision.
- **RLRT: Rebellious Student — Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR** [![arXiv](https://img.shields.io/badge/arXiv-2605.10781-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10781)
  - *(2026)* Reverses the self-distillation direction on correct rollouts: instead of pulling the student toward the privileged teacher (which overwrites self-driven reasoning), RLRT upweights tokens where the student diverged from the teacher and still reached the correct answer, augmenting GRPO with a "valuable exploration" signal grounded in already-successful student reasoning rather than uniform diversity; +8.9% average across six math benchmarks over self-distillation baselines on Qwen3-4B/8B base, instruct, and thinking checkpoints.

### Context and Experience Internalization

- **OPCD: On-Policy Context Distillation for Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2602.12275-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.12275)
  - *(2026)* Context-conditioned teacher on student rollouts; distills system prompts and experiential knowledge.
- **OEL: Online Experiential Learning for Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2603.16856-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.16856)
  - *(2026)* Deployment loop using OPCD for consolidating interaction traces into weights.
- **Aligning Language Models from User Interactions** [![arXiv](https://img.shields.io/badge/arXiv-2603.12273-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.12273)
  - *(2026)* Hindsight self-distillation from user follow-ups: student rolls out under the original prompt; same model conditioned on the user's follow-up serves as the token-level reverse-KL teacher.

### Efficiency Variants

- **Prefix OPD: Fast and Effective On-policy Distillation from Reasoning Prefixes** [![arXiv](https://img.shields.io/badge/arXiv-2602.15260-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.15260)
  - *(2026)* Distills only reasoning prefixes, cutting training FLOPs 2x-47x.
- **OVD: On-policy Verbal Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2601.21968-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.21968)
  - *(2026)* Trajectory-level verbal scoring instead of token-level logit matching; reduces memory and relaxes alignment requirements.
- **pi-Distill: Privileged Information Distillation for Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2602.04942-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.04942)
  - *(2026)* Training-time privileged information in agentic settings where only actions are observable.
- **Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2604.13010-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.13010)
  - *(2026)* Precomputes teacher log-probs once over SFT rollouts to eliminate the live-teacher server; 4x speedup via a teacher-consistency condition.
- **[Nitrobrew: Communication- and Memory-Efficient On-Policy Distillation](https://blog.tilderesearch.com/blog/nitrobrew)**
  - *(2026)* Systems-level OPD optimizations from Tilde Research: hidden-state teacher→student transport (~60x less bandwidth), tile-wise online divergence kernel (~37x less memory), and an SVD-compressed variant; 1.5-3x end-to-end throughput.
- **NPD: Near-Policy Distillation via Asynchronous Generation and Selective Packing** [![arXiv](https://img.shields.io/badge/arXiv-2605.05940-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.05940)
  - *(2026)* Decouples student generation from training and reframes OPD as SFT with sequence packing; sparse student updates plus Δ-IFD sample filtering keep updates inside a proximal learning zone; 8.1× speedup over on-policy baselines.
- **Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning** [![arXiv](https://img.shields.io/badge/arXiv-2605.07804-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.07804)
  - *(2026)* Monitors top-k student-teacher overlap to detect prefix drift; monotonically attenuates unreliable rewards and truncates drifted rollouts to reallocate compute toward locally exploitable teacher supervision.
- **EffOPD: Learning to Foresee — Unveiling the Unlocking Efficiency of On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2605.11739-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.11739)
  - *(2026)* Parameter-dynamics view of OPD's efficiency: early training establishes a "foresight" trajectory where updates concentrate on reasoning-critical modules (Module-Allocation) and dominant low-rank subspaces align with the final update direction (Update-Direction); EffOPD exploits this by adaptively extrapolating along the current update step for ~3× training acceleration with no extra trainables.

## Adjacent and Enabling Work

Papers that are not canonical OPD but matter for understanding or deploying it.

### Cross-Tokenizer and Model-Family Enablers

- **ULD: Towards Cross-Tokenizer Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2402.12030-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2402.12030)
  - *(2024)* Universal Logit Distillation; foundational enabler for cross-family OPD.
- **Multi-Level OT for Universal Cross-Tokenizer KD** [![arXiv](https://img.shields.io/badge/arXiv-2412.14528-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2412.14528)
  - *(2024)* Token- and sequence-level optimal transport for cross-tokenizer KD.
- **CDM: Enhancing Cross-Tokenizer KD with Contextual Dynamical Mapping** [![arXiv](https://img.shields.io/badge/arXiv-2502.11104-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.11104)
  - *(2025)* Contextual dynamic mapping for vocabulary alignment.
- **Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching** [![arXiv](https://img.shields.io/badge/arXiv-2503.20083-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.20083)
  - *(2025)* Approximate likelihood matching across fundamentally different tokenizers.
- **Cross-Tokenizer Likelihood Scoring Algorithms** [![arXiv](https://img.shields.io/badge/arXiv-2512.14954-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.14954)
  - *(2025)* Exact and approximate sequence likelihood scoring across BPE vocabularies.
- **DSKD: A Dual-Space Framework for General KD** [![arXiv](https://img.shields.io/badge/arXiv-2504.11426-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.11426)
  - *(2025)* Unifies output spaces; supports on- and off-policy KD between any two LLMs.
- **[GOLD: Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)**
  - *(2025)* Cross-tokenizer OPD with TRL integration.
- **CTPD: Cross Tokenizer Preference Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2601.11865-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.11865)
  - *(2026)* Aligned-span projection maps teacher and student tokens to shared character spans; teacher-anchored DPO with cross-tokenizer importance sampling for white-box preference transfer.
- **DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2602.21669-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.21669)
  - *(2026)* Dual-space token weighting plus Soft-DTW differentiable sequence alignment for cross-family teacher-student transfer.
- **Cross-Tokenizer LLM Distillation through a Byte-Level Interface** [![arXiv](https://img.shields.io/badge/arXiv-2604.07466-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.07466)
  - *(2026)* Converts teacher distributions to byte-level probabilities; adds a byte-level decoder to the student to enable distillation across mismatched tokenizers.
- **SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2605.07711-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.07711)
  - *(2026)* Replaces exact shared-token matching with short multi-token continuations that both tokenizers can realize; keeps the OPD loss form unchanged while recovering teacher signal at vocabulary-mismatched positions.

### Mismatch Mitigation and Student Quality

- **DistiLLM** [![arXiv](https://img.shields.io/badge/arXiv-2402.03898-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2402.03898)
  - *(2024)* Skew-KL with adaptive off-policy use of student-generated outputs.
- **Exploring and Enhancing Distribution Transfer in KD** [![arXiv](https://img.shields.io/badge/arXiv-2409.12512-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2409.12512)
  - *(2024)* Analyzes reverse-KL with student-generated output; proposes OKD.
- **FIRST: Efficient Trustworthy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2408.12168-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2408.12168)
  - *(2024)* Teacher recalibration for trustworthy offline KD.
- **Multi-Granularity Semantic Revision** [![arXiv](https://img.shields.io/badge/arXiv-2407.10068-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.10068)
  - *(2024)* Sequence correction for low-quality student-generated outputs.
- **Warmup-Distill** [![arXiv](https://img.shields.io/badge/arXiv-2502.11766-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.11766)
  - *(2025)* Bridges distribution mismatch before distillation begins.
- **TAID: Temporally Adaptive Interpolated Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2501.16937-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2501.16937)
  - *(2025)* Addresses teacher-student mismatch via adaptive interpolation.
- **DistiLLM-2** [![arXiv](https://img.shields.io/badge/arXiv-2503.07067-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.07067)
  - *(2025)* Contrastive extension; student-generated outputs collected per epoch.
- **SpecKD: Speculative Decoding for Effective KD** [![arXiv](https://img.shields.io/badge/arXiv-2510.24021-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.24021)
  - *(2025)* Speculative-decoding-inspired selective token-level losses.
- **Knowledge Distillation with Training Wheels** [![arXiv](https://img.shields.io/badge/arXiv-2502.17717-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.17717)
  - *(2025)* Entropy-regularized value optimization with on-/off-policy demonstrations.
- **Revealing the Power of Post-Training via KD** [![arXiv](https://img.shields.io/badge/arXiv-2509.26497-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.26497)
  - *(2025)* Offline on-policy KD: student generates, then teacher labels.
- **TSD-KD: Explain in Your Own Words** [![arXiv](https://img.shields.io/badge/arXiv-2603.13260-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.13260)
  - *(2026)* Student proposes candidates, teacher reranks, selective token distillation.
- **SSD: Embarrassingly Simple Self-Distillation Improves Code Generation** [![arXiv](https://img.shields.io/badge/arXiv-2604.01193-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.01193)
  - *(2026)* Temperature-shifted self-sampling plus SFT with no teacher or verifier; identifies precision-exploration conflict in token distributions.
- **AdaSwitch: Balancing Exploration and Guidance in KD via Adaptive Switching** [![arXiv](https://img.shields.io/badge/arXiv-2510.07842-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.07842)
  - *(2025)* Adaptively switches between on-policy student rollouts and off-policy teacher data using a context-aware divergence threshold.
- **DDT: Towards On-Policy SFT via Distribution Discriminant Theory** [![arXiv](https://img.shields.io/badge/arXiv-2602.12222-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.12222)
  - *(2026)* In-Distribution Finetuning and Hinted Decoding realign training data to the student's evolving distribution; matches offline RL at SFT cost.
- **DASD: Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning** [![arXiv](https://img.shields.io/badge/arXiv-2601.09088-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.09088)
  - *(2026)* On-policy correction pipeline addressing distribution mismatch, capacity misalignment, and exposure bias in sequence-level CoT distillation.
- **SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting** [![arXiv](https://img.shields.io/badge/arXiv-2604.10688-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.10688)
  - *(2026)* Routes correct rollouts to student-perplexity-weighted MLE and incorrect ones to teacher-perplexity-weighted KL distillation; per-prompt group-level normalization.
- **TIP: Token Importance in On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2604.14084-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.14084)
  - *(2026)* Selective token training on high-entropy positions and confidently-wrong low-entropy positions; matches full-token baselines at lower memory.
- **DP-OPD: Differentially Private On-Policy Distillation for Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2604.04461-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.04461)
  - *(2026)* Standard student-rollout + frozen-teacher OPD with DP-SGD on student updates; first OPD recipe compatible with sample-level differential privacy.
- **Distillation Traps and Guards: A Calibration Knob for LLM Distillability** [![arXiv](https://img.shields.io/badge/arXiv-2604.18963-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18963)
  - *(2026)* Diagnoses tail noise, off-policy instability, and teacher-student gap as KD/OPD failure modes; post-hoc calibrates teachers via RFT to control distillability.

### Preference, Reward-Guided, and Hybrid RL+KD

- **Direct Preference Knowledge Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2406.19774-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2406.19774)
  - *(2024)* Preference-aware KD combining reverse-KL with implicit reward objectives.
- **Online Knowledge Distillation with Reward Guidance** [![arXiv](https://img.shields.io/badge/arXiv-2505.18952-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.18952)
  - *(2025)* Sequential KD via preference optimization; offline and online variants.
- **KDRL** [![arXiv](https://img.shields.io/badge/arXiv-2506.02208-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.02208)
  - *(2025)* Unified reverse-KL KD with RL in a single post-training objective.
- **RLTF-SD: Expanding RL via Text Feedback** [![arXiv](https://img.shields.io/badge/arXiv-2602.02482-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.02482)
  - *(2026)* Internalizes text feedback via self-distillation.
- **RLAD: Reinforcement-aware KD for LLM Reasoning** [![arXiv](https://img.shields.io/badge/arXiv-2602.22495-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.22495)
  - *(2026)* Trust-region ratio distillation on student rollouts.
- **Multi-Token Prediction via Self-Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2602.06019-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.06019)
  - *(2026)* Online self-distillation for multi-token prediction and faster inference.
- **ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2509.25100-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.25100)
  - *(2025)* Mixed-policy teacher/student preference distillation using student-generated outputs; enables black-box cross-architecture transfer.
- **SRPO: Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing** [![arXiv](https://img.shields.io/badge/arXiv-2604.02288-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.02288)
  - *(2026)* Routes correct student rollouts to reward-based RL and failed ones to self-distillation; unifies GRPO and SDPO.
- **KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2504.19024-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.19024)
  - *(2025)* K-step return via Bellman equation replaces high-variance single-step REINFORCE in sequence-level OPD.
- **Rethinking LLM Distillation: A Constrained MDP Perspective** [![arXiv](https://img.shields.io/badge/arXiv-2509.22921-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.22921)
  - *(2025)* Maximizes task reward subject to hard KL constraint against the teacher; avoids manual Lagrangian tuning.
- **RLKD: Distilling LLMs' Reasoning via Reinforcement Learning** [![arXiv](https://img.shields.io/badge/arXiv-2505.16142-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.16142)
  - *(2025)* Generative Structure Reward Model captures multi-branch reasoning structure on student rollouts; outperforms SFT-RL pipelines on 0.1% data.
- **LUFFY: Learning to Reason under Off-Policy Guidance** [![arXiv](https://img.shields.io/badge/arXiv-2504.14945-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.14945)
  - *(2025)* Mixed-policy GRPO combining on-policy rollouts with off-policy teacher traces via regularized importance sampling.
- **BOND: Aligning LLMs with Best-of-N Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2407.14622-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.14622)
  - *(2024)* RL that mimics best-of-N sampling via Jeffreys-divergence distribution matching; eliminates inference-time BoN cost.
- **Faster WIND: Accelerating Iterative Best-of-N Distillation for LLM Alignment** [![arXiv](https://img.shields.io/badge/arXiv-2410.20727-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.20727)
  - *(2024)* Game-theoretic framing of iterative BoN as self-play; win-rate dominance optimization with sample-efficiency guarantees (AISTATS 2025).
- **AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2503.02832-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.02832)
  - *(2025)* Casts RLHF as token-level distillation by injecting DPO rewards; contrastive token-adaptive optimization (ACL 2025).
- **KEPO: Knowledge-Enhanced Preference Optimization for Reinforcement Learning with Reasoning** [![arXiv](https://img.shields.io/badge/arXiv-2602.00400-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.00400)
  - *(2026)* Quality-gated on-policy distillation on high-quality trajectories combined with knowledge-enhanced exploration via teacher hints.
- **𝒳-KD: General Experiential Knowledge Distillation for Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2602.12674-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.12674)
  - *(2026)* Jointly models the teacher's reward and performs policy distillation so the student learns inside the teacher's original learning environment.
- **ExGRPO: Probing to Refine — Reinforcement Distillation of LLMs via Explanatory Inversion** [![arXiv](https://img.shields.io/badge/arXiv-2603.19266-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.19266)
  - *(2026)* Explanatory probes plus dialogue-structure utility bonus reward coherent reasoning over memorized answers.
- **HPD: Hybrid Policy Distillation for LLMs** [![arXiv](https://img.shields.io/badge/arXiv-2604.20244-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.20244)
  - *(2026)* Unified reweighted-log-likelihood framework combining forward/reverse KL with off-policy and lightweight on-policy sampling.
- **NPO: Near-Future Policy Optimization** [![arXiv](https://img.shields.io/badge/arXiv-2604.20733-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.20733)
  - *(2026)* Uses a later checkpoint of the same policy as the teacher; AutoNPO adaptively triggers the switch to maximize learning signal in RLVR.
- **CoDistill-GRPO: A Co-Distillation Recipe for Efficient Group Relative Policy Optimization** [![arXiv](https://img.shields.io/badge/arXiv-2605.08873-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.08873)
  - *(2026)* Co-trains large and small models with dual GRPO objectives — the small model learns from the large model via an on-policy KD reward while the large model is updated only on importance-reweighted small-model rollouts; downsampling on a combined accuracy + KD score sharpens advantages, removing the frozen-oracle assumption and matching standard GRPO with an 18% training-time speedup.

### Self-Play and Iterative Bootstrapping

- **SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2401.01335-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2401.01335)
  - *(2024)* Self-play loop where the model distinguishes its own generations from human references; ICML 2024.
- **Self-Rewarding Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2401.10020-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2401.10020)
  - *(2024)* Iterative DPO with the model as LLM-as-Judge providing self-rewards on its own generations; supervision-equivalent feedback from same-model evaluation.
- **rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking** [![arXiv](https://img.shields.io/badge/arXiv-2501.04519-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2501.04519)
  - *(2025)* MCTS-guided self-evolution: policy and process reward model co-improve from scratch via code-augmented reasoning.
- **rStar2-Agent: Agentic Reasoning Technical Report** [![arXiv](https://img.shields.io/badge/arXiv-2508.20722-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.20722)
  - *(2025)* GRPO with Resample-on-Correct rollouts plus multi-stage SFT→RL recipe for a 14B agentic reasoner.
- **π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data** [![arXiv](https://img.shields.io/badge/arXiv-2604.14054-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.14054)
  - *(2026)* Examiner generates tasks plus question-construction-paths (QCPs); same-scale teacher uses QCPs as privileged context to densely supervise student rollouts via reverse KL; turns sparse-reward self-play into dense self-distillation.
- **SPHERE: Self-Evolved Preference Optimization for Enhancing Mathematical Reasoning in Small Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2503.04813-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.04813)
  - *(2025)* PRM/ORM-scored MCTS rollouts plus self-correction yield max-vs-min preference pairs for iterative DPO; bootstraps small reasoning models without external teachers.
- **SGS: Scaling Self-Play with Self-Guidance** [![arXiv](https://img.shields.io/badge/arXiv-2604.20209-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.20209)
  - *(2026)* Three-role self-play for formal theorem proving — Solver, Generator, and Reviewer share a base model; the Generator proposes simpler related problems trained with a reward combining the Solver's empirical solve rate (difficulty curriculum) and a Reviewer-scored relevance/cleanness signal that supervises against Generator-side reward hacking and collapse; surpasses the strongest RL baseline in fewer than 80 rounds and lets a 7B model out-solve a 671B model at pass@4 on Lean4 after 200 rounds.

### Agent Distillation, Multimodal, and Other Extensions

- **Structured Agent Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2505.13820-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.13820)
  - *(2025)* Queries teacher online to avoid distribution drift in agent settings.
- **From Deferral to Learning: Online In-Context KD for LLM Cascades** [![arXiv](https://img.shields.io/badge/arXiv-2509.22984-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.22984)
  - *(2025)* Teacher-student cascade with reusable online knowledge store.
- **AllMem** [![arXiv](https://img.shields.io/badge/arXiv-2602.13680-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.13680)
  - *(2026)* Offline on-policy distillation for long-context modeling.
- **Video-OPD** [![arXiv](https://img.shields.io/badge/arXiv-2602.02994-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.02994)
  - *(2026)* OPD for temporal video grounding in multimodal LLMs.
- **Reinforced Attention Learning** [![arXiv](https://img.shields.io/badge/arXiv-2602.04884-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.04884)
  - *(2026)* On-policy attention distillation for multimodal models.
- **SCoRe: From Correction to Mastery via Reinforced Distillation of LLM Agents** [![arXiv](https://img.shields.io/badge/arXiv-2509.14257-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.14257)
  - *(2025)* Student generates agent trajectories; teacher intervenes at first critical error for on-policy corrective distillation.
- **VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2510.23497-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.23497)
  - *(2025)* Text-only teacher distills reasoning into VLM student via student-generated traces with combined GRPO and OPD.
- **X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs** [![arXiv](https://img.shields.io/badge/arXiv-2603.24596-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.24596)
  - *(2026)* Student on-policy rollouts with token-level teacher feedback for cross-modal speech-LLM distillation.
- **VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2603.26666-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.26666)
  - *(2026)* Reverse-KL on-policy distillation bridging offline SFT and online RL for robotic manipulation.
- **TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents** [![arXiv](https://img.shields.io/badge/arXiv-2604.24005-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.24005)
  - *(2026)* Short-to-long trajectory-depth curriculum that mitigates trajectory-level KL instability when accumulated multi-turn errors push student behavior beyond teacher support.
- **LLM4Teach: Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents** [![arXiv](https://img.shields.io/badge/arXiv-2311.13373-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2311.13373)
  - *(2023)* LLM teacher distills into a small RL agent that surpasses the teacher through environment interaction.
- **RPD: Refined Policy Distillation — From VLA Generalists to RL Experts** [![arXiv](https://img.shields.io/badge/arXiv-2503.05833-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.05833)
  - *(2025)* Teacher VLA actions guide the student policy during RL exploration; combines RL with behavioral cloning (IROS 2026).
- **π-Flow: Policy-Based Few-Step Generation via Imitation Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2510.14974-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.14974)
  - *(2025)* Imitation distillation aligns student flow-model policy trajectories with teacher trajectories under standard flow matching loss (ICLR 2026).
- **Step-Audio-R1 Technical Report** [![arXiv](https://img.shields.io/badge/arXiv-2511.15848-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.15848)
  - *(2025)* Modality-Grounded Reasoning Distillation produces audio reasoning chains grounded in acoustic features rather than hallucinated text.
- **OPD-AVMP: On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning** [![arXiv](https://img.shields.io/badge/arXiv-2604.07944-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.07944)
  - *(2026)* Generalized on-policy KD for LLM-based driving planners; 5x compression at near-teacher performance.
- **CORD: Bridging the Audio–Text Reasoning Gap via Weighted On-policy Cross-modal Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2601.16547-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.16547)
  - *(2026)* Audio-conditioned student rollouts; text-conditioned same model as in-model teacher; importance-weighted reverse KL on early/critical tokens plus GRPO at sequence level.
- **Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents** [![arXiv](https://img.shields.io/badge/arXiv-2604.10674-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.10674)
  - *(2026)* Student rolls out under plain prompt; same model under skill-augmented prompt serves as token-level self-teacher for multi-turn agent training.
- **CoPD: Co-Evolving Policy Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2604.27083-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.27083)
  - *(2026)* Parallel expert training with bidirectional OPD where experts co-evolve as mutual teachers during RLVR; integrates text, image, and video reasoning while avoiding both mixed-RLVR inter-capability divergence and sequential-MOPD behavioral gaps.
- **PRISM: Beyond SFT-to-RL — Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL** [![arXiv](https://img.shields.io/badge/arXiv-2604.28123-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.28123)
  - *(2026)* Inserts a black-box OPD distribution-alignment stage between SFT and RLVR for VLMs; MoE discriminator with disentangled perception and reasoning experts provides response-level adversarial corrective signals on student rollouts without teacher logits.
- **D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models** [![arXiv](https://img.shields.io/badge/arXiv-2605.05204-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.05204)
  - *(2026)* Ports OPSD from LLMs to few-step T2I diffusion models; same model acts as student (text-only conditioning) and teacher (text+target-image multimodal conditioning), with velocity-MSE distillation on the student's own few-step rollouts to learn new concepts/styles without breaking distilled few-step inference.
- **Flow-OPD: On-Policy Distillation for Flow Matching Models** [![arXiv](https://img.shields.io/badge/arXiv-2605.08063-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.08063)
  - *(2026)* Multi-teacher OPD for text-to-image flow matching: per-domain Flow-GRPO experts supply dense reverse-KL trajectory-level supervision on the student's on-policy SDE rollouts via hard task routing, with Manifold Anchor Regularization from a frozen aesthetic teacher preventing the reward hacking typical of scalar-reward joint training.
- **VISD: Enhancing Video Reasoning via Structured Self-Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2605.06094-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.06094)
  - *(2026)* Video-aware judge decomposes reasoning quality into structured dimensions; an EMA teacher conditioned on this privileged feedback supervises student rollouts at the token level, with direction-magnitude decoupling to stably integrate dense supervision with RL.
- **TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM** [![arXiv](https://img.shields.io/badge/arXiv-2605.09536-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.09536) [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/BHmingyang/TAD)
  - *(2026)* Adapts self-distillation to diffusion LLMs: a teacher conditioned on prompt + ground-truth response records masked-decoding trajectories, partitioning masked positions into near (hard cross-entropy) and distant (soft KL) subsets by remaining decoding steps; yields paired Quality and Speed deployment models that improve LLaDA's accuracy-parallelism trade-off (avg accuracy 46.2 → 51.6 for Quality; avg AUP 46.2 → 257.1 for Speed).
- **DiMO: Distilling Masked Diffusion Models into One-step Generator** [![arXiv](https://img.shields.io/badge/arXiv-2503.15457-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.15457)
  - *(2025)* First on-policy distillation for masked discrete diffusion image generation (class-conditional and text-to-image from Meissonic teacher): student-induced intermediate states $\tilde{x}_t$ from forward-masking the one-step student's outputs drive token-level distribution matching against the frozen teacher, with a DMD-style auxiliary model approximating the student distribution to make the gradient tractable; Generalized Jeffrey divergence (FKL/RKL mix) avoids RKL mode-seeking, and a hybrid mask/random-token initialization with Gaussian embedding perturbation prevents one-step mode collapse from the deterministic all-mask init (ICCV 2025).

### Speculative Decoding (Draft-Model Training)

Draft-model training for speculative decoding shares OPD's core loop: the draft (student) generates, the target (teacher) verifies, and the draft is updated to match. Included for breadth even though the goal is inference acceleration rather than student capability.

- **Online Speculative Decoding** [![arXiv](https://img.shields.io/badge/arXiv-2310.07177-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2310.07177)
  - *(2023)* Continuously updates the draft model on observed user queries via knowledge distillation; 1.42x-2.17x latency gains.
- **DistillSpec: Improving Speculative Decoding via Knowledge Distillation** [![arXiv](https://img.shields.io/badge/arXiv-2310.08461-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2310.08461)
  - *(2023)* Aligns draft with target via on-policy data and task-tailored divergence; foundational draft-model OPD recipe (ICLR 2024).
- **HASS: Learning Harmonized Representations for Speculative Sampling** [![arXiv](https://img.shields.io/badge/arXiv-2408.15766-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2408.15766)
  - *(2024)* Harmonized objective and context distillation to fix train-decoding inconsistency in speculative sampling.
- **Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting** [![arXiv](https://img.shields.io/badge/arXiv-2412.12639-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2412.12639)
  - *(2024)* Coupled Sequential Glancing Distillation strengthens inter-token dependencies in semi-autoregressive drafters.
- **CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter** [![arXiv](https://img.shields.io/badge/arXiv-2502.16880-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.16880)
  - *(2025)* Cross-step representation alignment for multi-step drafter training (ACL 2025).
- **EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test** [![arXiv](https://img.shields.io/badge/arXiv-2503.01840-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.01840)
  - *(2025)* Direct token prediction with multi-layer feature fusion under on-policy training-time test; up to 6.5x speedup.
- **MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2505.10526-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.10526)
  - *(2025)* Adapts an SLM into a VLM drafter via self-distilled visual instruction tuning.
- **DVI: Draft, Verify, and Improve — Toward Training-Aware Speculative Decoding** [![arXiv](https://img.shields.io/badge/arXiv-2510.05421-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.05421)
  - *(2025)* Self-speculative drafter trained online from verifier decisions via KL→RL schedule with reward-masked cross-entropy.
- **ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems** [![arXiv](https://img.shields.io/badge/arXiv-2510.26475-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.26475)
  - *(2025)* Evolves the drafter during RL via reward-weighted distillation on rollouts.
- **[DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution](https://openreview.net/forum?id=CRgWv0kWjF)**
  - *(2026)* Multimodal speculative-reasoning drafter trained for faithfulness to target trajectories with verifier-gated parallel execution.
- **MSD: Speculative Decoding Reimagined for Multimodal Large Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2505.14260-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.14260)
  - *(2025)* Decouples text and visual tokens in the draft model; two-stage training (text-only then progressively multimodal) lifts MLLM speculative speedups to 2.29–2.46x.
- **SpecVLM: Fast Speculative Decoding in Vision-Language Models** [![arXiv](https://img.shields.io/badge/arXiv-2509.11815-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.11815)
  - *(2025)* Elastic visual compressor plus online-logit distillation trains the draft model from on-the-fly teacher logits and features, eliminating offline corpora; 2.5–2.9× end-to-end VLM speedups.
- **ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding** [![arXiv](https://img.shields.io/badge/arXiv-2509.15235-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.15235)
  - *(2025)* Lightweight vision adaptor compresses image tokens for the draft; trained on target-generated long responses while preventing hidden-state shortcut learning.
- **Aurora: When RL Meets Adaptive Speculative Training — A Unified Training-Serving System** [![arXiv](https://img.shields.io/badge/arXiv-2602.06932-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.06932)
  - *(2026)* Online continual draft training: target verifications on draft proposals stream into FKL/RKL fine-tuning of the speculator, then hot-swap back into serving; closes the train-serve loop.
- **SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting** [![arXiv](https://img.shields.io/badge/arXiv-2605.07243-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.07243)
  - *(2026)* Block-iterative drafter with layer-wise shift for in-block dependence and inheritable hidden states across blocks; valid-prefix masking, co-trained rank head, and cost-aware bandit adaptation from verifier feedback.
- **SFDD: Flatter Tokens are More Valuable for Speculative Draft Model Training** [![arXiv](https://img.shields.io/badge/arXiv-2601.18902-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.18902)
  - *(2026)* Sample-level flatness filters EAGLE training data to the most acceptance-valuable samples; 2× training speedup at 50% data with <4% inference-speedup loss.

### Precursors

- **Autoregressive KD through Imitation Learning** [![arXiv](https://img.shields.io/badge/arXiv-2009.07253-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2009.07253)
  - *(2020)* Early precursor framing sequence-model KD as imitation learning.
- **Learning by Distilling Context** [![arXiv](https://img.shields.io/badge/arXiv-2209.15189-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2209.15189)
  - *(2022)* Context distillation; key precursor to OPCD and OEL.

## Technical Reports and Industrial Recipes

Production training pipelines that use OPD as a post-training stage.

| Year | System | OPD Usage | Link |
|------|--------|-----------|------|
| 2024 | Gemma 2 | KD as alternative to next-token prediction for the 2B and 9B student models | [![arXiv](https://img.shields.io/badge/arXiv-2408.00118-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2408.00118) |
| 2025 | Qwen3 | Strong-to-weak; off-policy then on-policy distillation | [![arXiv](https://img.shields.io/badge/arXiv-2505.09388-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.09388) |
| 2025 | Qwen3-Omni | Off-policy then on-policy distillation before GSPO | [![arXiv](https://img.shields.io/badge/arXiv-2509.17765-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.17765) |
| 2025 | GLM-4.5 / 4.6 | Multi-stage post-training with expert model iteration and RL (precursor to GLM-5's explicit OPD) | [![arXiv](https://img.shields.io/badge/arXiv-2508.06471-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.06471) |
| 2025 | HY-MT1.5 | Multi-stage translation: SFT + OPD + RL | [![arXiv](https://img.shields.io/badge/arXiv-2512.24092-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.24092) |
| 2026 | MiMo-V2-Flash | Multi-Teacher OPD (MOPD) as post-training stage | [![arXiv](https://img.shields.io/badge/arXiv-2601.02780-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.02780) |
| 2026 | GLM-5 | On-policy cross-stage distillation to recover earlier skills | [![arXiv](https://img.shields.io/badge/arXiv-2602.15763-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.15763) |
| 2026 | Typhoon-S | Minimal sovereign recipe: SFT + OPD + small-scale RFT | [![arXiv](https://img.shields.io/badge/arXiv-2601.18129-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.18129) |
| 2026 | Nemotron-Cascade 2 | Cascade RL + multi-domain on-policy distillation | [![arXiv](https://img.shields.io/badge/arXiv-2603.19220-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.19220) |
| 2026 | Baichuan-M3 | Three-stage: task RL, offline policy distillation, multi-teacher OPD | [![arXiv](https://img.shields.io/badge/arXiv-2602.06570-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.06570) |
| 2026 | MobileLLM-R1.5 | Final-stage on-policy KD as primary improvement over R1 | model card |
| 2026 | Nanbeige4-3B-Thinking | OPD preferred over off-policy for math reasoning | [model card](https://huggingface.co/Nanbeige/Nanbeige4-3B-Thinking-2510) |
| 2026 | DeepSeek-V4 | Two-stage post-training: domain-expert SFT+GRPO, then unified model consolidation via on-policy distillation | [report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| 2026 | Qwen3.5-Omni | Specialist teacher distillation, then privileged-input self-distillation aligning audio-conditioned outputs to text-conditioned responses (labeled OPD by the report) | [![arXiv](https://img.shields.io/badge/arXiv-2604.15804-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.15804) |
| 2026 | HY-Embodied-0.5 | Large-to-small on-policy distillation transfers 32B embodied-reasoning behavior into the 2B edge variant: student rollouts, teacher token-level supervision at the same prefixes | [![arXiv](https://img.shields.io/badge/arXiv-2604.07430-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.07430) |
| 2026 | KAT-Coder-V2 | Specialize-then-Unify: five domain-expert coding agents (SWE, WebCoding, Terminal, WebSearch, General) each trained with SFT+RL, then consolidated into one model via on-policy distillation on student trajectories | [![arXiv](https://img.shields.io/badge/arXiv-2603.27703-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.27703) |

## Frameworks, Tools, and Implementations

### Training Frameworks

| Framework | Description | Link |
|---|---|---|
| TRL | GKD, GOLD, and MiniLLM trainers; most accessible starting point | [docs](https://huggingface.co/docs/trl) |
| NeMo-RL | Multi-teacher and cross-tokenizer OPD at scale | [docs](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html), [repo](https://github.com/NVIDIA-NeMo/RL) |
| veRL | Async on-policy KD trading strict on-policy guarantees for throughput | [docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) |
| MS-Swift | GKD and OPSD sections in the ModelScope ecosystem | [docs](https://swift.readthedocs.io/en/latest/) |
| EasyDistill | Comprehensive KD toolkit for black-box and white-box LLM distillation | [![arXiv](https://img.shields.io/badge/arXiv-2505.20888-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.20888) |
| KDFlow | Off-policy, on-policy, and cross-tokenizer distillation via decoupled backends | [![arXiv](https://img.shields.io/badge/arXiv-2603.01875-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.01875), [repo](https://github.com/songmzhang/KDFlow) |
| slime | Unified RL stack supporting on-policy distillation and hindsight hints | [repo](https://github.com/) |
| OpenClaw-RL | Agentic RL stack with hindsight-guided OPD | [![arXiv](https://img.shields.io/badge/arXiv-2603.10165-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.10165) |
| NexRL | Dedicated on-policy distillation recipes | [repo](https://github.com/nex-agi/NexRL) |
| SkyRL | OPD examples and blog resources | [repo](https://github.com/NovaSky-AI/SkyRL) |
| ATLAS | Continual-learning framework using GKD/GRPO from runtime traces | [docs](https://docs.arc.computer/introduction) |
| AReaL | OPD and KDRL implementation over student-sampled trajectories with teacher log-prob guidance | [docs](https://github.com/inclusionAI/AReaL/blob/main/docs/en/algorithms/distillation.md) |
| SpecForge | Open-source training framework for speculative draft models with EAGLE-3 support, target-draft decoupling, and hybrid parallelism | [![arXiv](https://img.shields.io/badge/arXiv-2603.18567-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.18567), [repo](https://github.com/sgl-project/SpecForge) |
| Tinker Cookbook | Thinking Machines' Tinker SDK recipes for off-policy KD plus single-teacher, multi-teacher, and multi-turn tool-use on-policy distillation | [recipes](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation), [repo](https://github.com/thinking-machines-lab/tinker-cookbook) |
| ROLL | Alibaba's scalable RL library for LLMs/VLMs with an on-policy distill pipeline alongside RL post-training | [repo](https://github.com/alibaba/ROLL) |

### Implementations

- [OPSD](https://github.com/siyan-zhao/OPSD) — Official code for Self-Distilled Reasoner / OPSD.
- [SCOPE](https://github.com/machine981/SCOPE) — Official implementation of SCOPE's dual-path OPD: student-PPL-weighted MLE for correct rollouts, teacher-PPL-weighted KL for incorrect.
- [CaOPD](https://github.com/SalesforceAIResearch/CaOPD) — Official implementation of CaOPD: K student rollouts → empirical success rate → confidence target replacement → standard reverse-KL OPD.
- [OPSD-OnPolicyDistillation](https://github.com/HJSang/OPSD_OnPolicyDistillation) — verl-based OPD implementation with separate-teacher distillation, agent-loop rollouts, and memory-efficient teacher/student execution.
- [nano-opd](https://github.com/Athe-kunal/nano-opd) — Hackable OPD library decoupling vLLM rollout, FSDP student training, and teacher forward passes across independent GPU groups; configurable reverse-KL / top-k objectives, weight-sync backend, and chunked long-sequence handling for easy swapping of models, data, and training knobs.

### Essays, Blog Posts, and Walkthroughs

- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (2025) — Best single-article introduction. Covers concepts, intuition, and practical use cases.
- [Unlocking On-Policy Distillation for Any Model Family (GOLD)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) (2025) — Cross-tokenizer OPD walkthrough with TRL code.
- [On SFT, RL, and on-policy distillation](https://x.com/willccbb/status/2050038277454143918) (2026) — Will Brown's essay framing OPD via the SFT-vs-RL compounding argument and gradient geometry (sparse/dense × biased/unbiased × concentrated/diffuse), with directions toward an optimal teacher on the reward-vs-KL Pareto curve.
- [OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践](https://zhuanlan.zhihu.com/p/2033212181823608430) (2026) — Chinese-language Zhihu deep-dive deriving OPD's sequence- and token-level reverse-KL objectives, comparing sampled-token / top-k / full-vocab signals and the k1/k2/k3 KL estimators, and mapping each variant to MiniLLM, GKD/SWIFT, verl, and DeepSeek V4's full-vocab multi-teacher recipe.
- [Multi-Teacher On-Policy Distillation: A New Post-Training Primitive](https://yumoxu.notion.site/multi-teacher-on-policy-distillation) (2026) — Yumo Xu's essay surveying the emergence of multi-teacher OPD (MOPD) as a post-training primitive: derives the OPD loss as GRPO with the advantage replaced by stop-gradient teacher/student log-ratio, then organizes recent MOPD deployments into final-stage consolidation (MiMo-V2-Flash, GLM-5), mid-pipeline forgetting recovery (Nemotron-Cascade 2), and scaled-up regimes (DeepSeek-V4) with full-vocabulary logits, 10+ teachers, and dedicated infrastructure for teacher scheduling and fault-tolerant rollouts.
- [On-Policy Distillation: Theory & Practice in Model Merging](https://www.notion.so/On-Policy-Distillation-Theory-Practice-in-Model-Merging-2f44795a3e8b801cbedee2c96a23c788) (2026) — ByteDance Seed blog framing OPD as entropy-regularized RL, contrasting REINFORCE's unbiased accumulated log-ratio-to-go against the biased but lower-variance per-token log ratio, with case studies on cross-tokenizer teacher-sequence pitfalls and OPD reward hacking during agent model merging.
- [Distilling 100B+ Models 40x Faster with TRL](https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer) (2026) — Hugging Face engineering walkthrough of TRL's `DistillationTrainer` scaling tricks: a generation buffer that batches student rollouts across gradient-accumulation steps without breaking on-policy, plus request batching and base64-encoded NumPy logprob payloads on the teacher vLLM server, yielding ~40x speedup and validated on a Qwen3-235B → Qwen3-4B math run.
- [SFT, RL, and On-Policy Distillation Through a Distributional Lens](https://x.com/nrehiew_/status/2053482349300797526) (2026) — wh's distributional-geometry framing of SFT (forward-KL pull to an external target), RL (reward direction with on-policy data implicitly bounding KL to the starting policy), and OPD (RL-like on-policy state distribution with teacher-derived per-token signal); a Minimal Code Editing experiment shows OPD students from an SFT teacher and from an RL teacher converge to nearly identical performance and forget less than even the SFT teacher itself, arguing on-policy data — not the teacher choice or explicit KL penalties — is the load-bearing anti-forgetting ingredient.
- [What Apple found out about On-Policy Distillation](https://x.com/neural_avb/status/2054585001757614172) (2026) — AVB's tutorial-style breakdown of Apple's "Unmasking OPD" paper (arXiv 2605.10889), walking through the training-free gradient-alignment framework that measures whether a teacher's nudge will help or hurt the student before any training, and summarizing the paper's findings — best-teacher choice flips with student size and task difficulty, token-level features (KL, entropy, depth) only weakly predict alignment, gradient alignment oscillates token-by-token within a single trace, wrong demonstrations hurt self-distillation except on hard math, and summarized demos help larger students but not smaller ones.

## Acknowledgments

This list draws on the parallel curation effort at [thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD), which provided pointers to several papers (notably the speculative-decoding draft-model training, BoN distillation, self-play, and additional industrial reports) that broadened the scope of this list. The two lists are organized differently — thinkwee/AwesomeOPD groups by feedback signal and access mode; this list groups by methodological role — and are best read together.

## Contributing

Contributions welcome! Please open a PR if you know of papers, reports, or tools related to on-policy distillation. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed criteria, section placement, and formatting guidelines.

- **Inclusion criteria:** The work should involve student rollouts as central to the learning signal, or directly enable OPD deployment (cross-tokenizer, frameworks, etc.).
- **Entry format:** Arxiv papers use a bold title with an arXiv badge, then a second-line `*(Year)* One-line description.` Non-arxiv items (blog posts, repos) use `[Title](url) (Year) — One-line description.` See [CONTRIBUTING.md](CONTRIBUTING.md) for full examples.

## Citation

If you find this resource useful, please cite it as:

```bibtex
@software{awesome-on-policy-distillation,
  title = {{Awesome On-Policy Distillation}},
  author = {Liu, Chris Yuhao and others},
  year = {2026},
  doi = {10.5281/zenodo.19411493},
  url = {https://github.com/chrisliu298/awesome-on-policy-distillation},
  version = {v1.0.0}
}
```

---

*Last updated: 2026-05-15. Coverage: core OPD papers, adjacent work (including speculative-decoding draft-model training and self-play / iterative bootstrapping), surveys, technical reports, and tooling through May 2026.*
