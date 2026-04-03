# Safety Head Attribution with nnsight

An [nnsight](https://nnsight.net/) reimplementation of **"On the Role of Attention Heads in Large Language Model Safety"** ([arXiv 2410.13708](https://arxiv.org/abs/2410.13708)), running on [NDIF](https://ndif.us/) for remote model execution.

Original repository: [ydyjya/SafetyHeadAttribution](https://github.com/ydyjya/safetyheadattribution)

## What this does

Safety-aligned LLMs can be "broken" by ablating a tiny fraction of their attention heads. This notebook identifies those safety-critical heads in **Llama-2-7b-chat-hf** using three methods from the paper:

1. **SHIPS** (Safety Head ImPortant Score) — masks individual attention heads and measures KL divergence from baseline to find which heads are most responsible for safety behavior on a given query.
2. **Sahara** (Safety Attention Head AttRibution Algorithm) — uses SVD subspace similarity across a dataset of harmful prompts to identify heads that consistently shift model behavior toward refusal.
3. **Surgery** — ablates the top safety heads identified by Sahara and evaluates how many harmful queries the model now answers (vs. baseline refusal rate).

Key finding from the paper: ablating a single safety head (0.006% of parameters) allows aligned models to respond to 16x more harmful queries.

## Implementation

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Framework | Custom PyTorch hooks | nnsight (NDIF remote execution) |
| Compute | Local GPU | Remote via NDIF — no local GPU required |
| SVD dtype | Direct | `.float()` conversion for SVD compatibility |

The notebook includes a `QUICK_MODE` toggle for fast testing (~2-5 min) vs. full replication.

## Datasets

Harmful query datasets used for evaluation (all from established benchmarks):

- `maliciousinstruct.csv` — [MaliciousInstruct](https://github.com/Princeton-SysML/Jailbreak_LLM)
- `advbench.csv` — [AdvBench](https://github.com/llm-attacks/llm-attacks)
- `harmful_behaviors.csv` / `data_harmful-behaviors.csv` — HarmBench harmful behaviors
- `jailbreakbench.csv` — [JailbreakBench](https://jailbreakbench.github.io/)

## Usage

1. Get an [NDIF API key](https://login.ndif.us/) and a [HuggingFace token](https://huggingface.co/settings/tokens) (with Llama 2 access)
2. Open `demo.ipynb` and run cells in order
3. Set `QUICK_MODE = False` for full analysis matching the paper
