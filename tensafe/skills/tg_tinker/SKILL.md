# TG Tinker (Agentic LoRA Creator)

## Description

TG Tinker creates production-quality LoRA adapters for anyone -- not just TenSafe
users. It ingests any dataset (PDF, CSV, JSONL, TXT, Parquet), auto-detects domain
and format, curates training data, runs SFT with differential privacy, screens
through RVUv2 safety layers, and packages the result as a signed TGSP file ready
for the TenSafe marketplace.

Default output is **TGSP** (safe, signed, sellable). Optional export to plain
LoRA formats: safetensors, PyTorch (.pt/.bin), and GGUF.

---

## Modes

### Mode 1: One-Click Creation

End-to-end adapter creation from a data directory.

```
tensafe train --data ./my_corpus --rank 30 --alpha 64
```

Pipeline steps:
1. `analyze_data(data_path)` -- detect format, volume, domain
2. `classify_domain(analysis)` -- auto-detect if not specified
3. `curate_dataset(data_path)` -- clean, extract Q&A pairs, split 80/20
4. `train_sft(dataset, lora_config)` -- SFT with DP, rank 30, alpha 64
5. `evaluate(checkpoint, test_split)` -- qa_verify score
6. `screen_rvu(checkpoint)` -- RVUv2 3-layer safety screening
7. `package_tgsp(checkpoint, metadata, signing_key)` -- create .tgsp
8. `export(tgsp_path, format)` -- optional: safetensors / pytorch / gguf

### Mode 2: Agent Self-Improvement

An agent reviews its own interaction logs and feedback, identifies weak spots,
curates targeted training data, and produces an improved adapter version.

```python
tinker.improve_adapter(current_adapter, interaction_log, feedback)
```

### Mode 3: Continuous Improvement Loop

Runs in the background, sampling live traffic, scoring responses, extracting
low-confidence examples, and periodically producing updated adapter versions.

```python
tinker.continuous_improve(adapter_path, traffic_sample, loop_config)
```

---

## Triggers

The following natural-language triggers activate TG Tinker:

- "Create a LoRA"
- "Create a TGSP adapter"
- "Turn this into a LoRA"
- "Improve this adapter"
- "Train an adapter from this data"
- "Build a LoRA for [domain]"
- "Fine-tune on this dataset"

---

## Quality Gates

Every adapter must pass these gates before packaging:

| Gate             | Threshold   | Description                                      |
|------------------|-------------|--------------------------------------------------|
| `qa_verify`      | >= 0.80     | QA score on held-out test split                  |
| `rvu_v2_pass`    | pass        | RVUv2 3-layer safety screening (toxicity, bias, IP) |
| `dp_budget`      | check       | Differential privacy epsilon budget verification |

If any gate fails, TG Tinker reports the failure, suggests remediation, and
does not produce a .tgsp file until the issue is resolved.

---

## LoraConfig Defaults

| Parameter        | Default Value                                             |
|------------------|-----------------------------------------------------------|
| `rank`           | 30                                                        |
| `alpha`          | 64                                                        |
| `dropout`        | 0.0                                                       |
| `target_modules` | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` |
| `bias`           | none                                                      |
| `task_type`      | CAUSAL_LM                                                 |

These defaults produce adapters compatible with Qwen 2.5 and Llama-family
architectures. Override via CLI flags or the `LoraConfig` dataclass.

---

## CLI Commands

| Command                                         | Description                     |
|--------------------------------------------------|---------------------------------|
| `tensafe init --domain DOMAIN --name NAME`       | Scaffold a new adapter project  |
| `tensafe train --data PATH --rank 30 --alpha 64` | Train adapter from data         |
| `tensafe screen PATH`                            | Run RVUv2 safety screening      |
| `tensafe pack --sign --creator "Name <email>"`   | Package as signed TGSP          |
| `tensafe verify FILE.tgsp`                       | Verify TGSP integrity           |
| `tensafe export FILE.tgsp --format safetensors`  | Export to plain LoRA format     |
| `tensafe import LORA_DIR --sign --name NAME`     | Import existing LoRA to TGSP    |
| `tensafe publish FILE.tgsp --marketplace`        | Publish to marketplace          |
| `tensafe validate FILE.tgsp`                     | Run TenSafe validation suite    |

---

## Output Formats

| Format       | Extension      | Description                                    |
|--------------|----------------|------------------------------------------------|
| TGSP         | `.tgsp`        | TenSafe signed package (default, marketplace-ready) |
| safetensors  | `.safetensors` | HuggingFace safetensors format                 |
| PyTorch      | `.pt` / `.bin` | Standard PyTorch state dict                    |
| GGUF         | `.gguf`        | GGML universal format for llama.cpp            |

---

## Every TGSP is a SKILL.md

Each .tgsp adapter contains an embedded markdown document (`skill_doc` field in the
manifest) that describes what the adapter does, when to use it, its capabilities,
and how it composes with other adapters. This means every TGSP file IS a skill file
that any AI agent can read to decide whether to load it.

When TG Tinker creates an adapter, it auto-generates the embedded SKILL.md based on:
- The training data domain and content
- The adapter's evaluated capabilities (from qa_verify)
- The LoraConfig and model architecture
- Compliance tags (HIPAA, PCI DSS, etc.)

The creator can also provide a custom skill_doc during creation:

```bash
tensafe pack --sign --skill-doc ./my_skill_description.md
```

---

## Adapter Swap Red-Teaming

**Every adapter swap — linear or non-linear — must be red-teamed and confirmed
before it takes effect.** This is a non-negotiable safety requirement.

When an agent (or self-improvement loop) wants to swap to a new adapter:

1. The new adapter passes the 7-step TGSP Load Gate (hash, signatures, RVUv2, etc.)
2. The meta-agent (or a human-in-the-loop) runs red-team checks:
   - Compare outputs of old vs new adapter on a held-out test set
   - Check for regression on safety-critical prompts
   - Verify the new adapter doesn't introduce bias or harmful outputs
   - Confirm the swap is justified (quality improvement ≥ 2%)
3. Only after approval does the swap execute (atomically)
4. If rejected, the swap is blocked and an incident report is generated

Swap approval modes:
- `meta_agent` — the meta-agent performs automated red-teaming and approves/rejects
- `human_in_loop` — a human must explicitly approve every swap
- Configurable in Helm chart: `agents.swapApproval: "meta_agent"` or `"human_in_loop"`

---

## Architecture

```
tensafe/skills/tg_tinker/
  SKILL.md           -- this file
  __init__.py        -- package init, exports TGTinker
  tinker.py          -- core agentic LoRA creator
  export.py          -- TGSP <-> LoRA format conversion
  data_analyzer.py   -- data format detection and domain classification
```
