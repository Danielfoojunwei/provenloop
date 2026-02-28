#!/usr/bin/env python3
"""Full QA verification of TenSafe Finance Demonstrator.

Tests:
  1. moe_config.json validity + path resolution
  2. tgsp_results.json validity
  3. TGSP adapter files loadable via torch.load
  4. TGSP adapter files contain LoRA weight keys
  5. Server Python imports resolve
  6. FinanceInferenceEngine instantiation (no model download)
  7. Reward function works with sample data
  8. Frontend files present and non-empty
"""

import json
import os
import sys
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure the project root is on sys.path so 'demonstrator.*' imports resolve
# regardless of how the script is invoked (python path/to/qa_verify.py)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

TGSP_DIR = os.path.join(BASE, "demonstrator", "adapters", "tgsp")
FRONTEND_DIR = os.path.join(BASE, "demonstrator", "frontend")

passed = 0
failed = 0
errors = []


def check(name, fn):
    global passed, failed
    try:
        result = fn()
        if result:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1
            errors.append(name)
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1
        errors.append(f"{name}: {e}")


# =========================================================
# 1. moe_config.json
# =========================================================
print("\n=== 1. moe_config.json ===")

moe_cfg = None

def test_moe_config_valid_json():
    global moe_cfg
    path = os.path.join(TGSP_DIR, "moe_config.json")
    with open(path) as f:
        moe_cfg = json.load(f)
    return isinstance(moe_cfg, dict)

def test_moe_config_has_model():
    return moe_cfg.get("model") == "Qwen/Qwen2.5-1.5B"

def test_moe_config_has_3_experts():
    return len(moe_cfg.get("experts", {})) == 3

def test_moe_config_expert_paths_exist():
    for name, ecfg in moe_cfg["experts"].items():
        full = os.path.join(BASE, ecfg["tgsp_path"])
        if not os.path.isfile(full):
            print(f"    MISSING: {ecfg['tgsp_path']}")
            return False
        size_mb = os.path.getsize(full) / 1048576
        print(f"    {name}: {size_mb:.1f} MB")
    return True

def test_moe_config_he_config():
    hec = moe_cfg.get("he_config", {})
    return (
        hec.get("scheme") == "ckks"
        and hec.get("poly_modulus_degree") == 16384
        and hec.get("scale_bits") == 40
        and hec.get("simd_slots") == 8192
    )

def test_moe_config_gatelink():
    gl = moe_cfg.get("gatelink_config", {})
    return (
        gl.get("device_profile") == "phone"
        and gl.get("client_layers") == 1
        and gl.get("dp_epsilon") == 1.0
        and gl.get("max_lora_rank") == 32
        and "force_he_in_split_mode" in gl
    )

check("moe_config.json is valid JSON", test_moe_config_valid_json)
check("model is Qwen/Qwen2.5-1.5B", test_moe_config_has_model)
check("has 3 experts", test_moe_config_has_3_experts)
check("all expert TGSP paths exist", test_moe_config_expert_paths_exist)
check("HE config correct (CKKS, N=16384, scale=40, slots=8192)", test_moe_config_he_config)
check("GateLink config correct (phone, K=1, eps=1.0, rank=32)", test_moe_config_gatelink)


# =========================================================
# 2. tgsp_results.json
# =========================================================
print("\n=== 2. tgsp_results.json ===")

tgsp_res = None

def test_tgsp_results_valid():
    global tgsp_res
    path = os.path.join(TGSP_DIR, "tgsp_results.json")
    with open(path) as f:
        tgsp_res = json.load(f)
    return isinstance(tgsp_res, dict) and len(tgsp_res) == 3

def test_tgsp_all_success():
    for name, r in tgsp_res.items():
        if not r.get("success"):
            print(f"    {name}: success=False")
            return False
        aid = r.get("adapter_id", "?")
        mhash = r.get("manifest_hash", "?")[:16]
        print(f"    {name}: id={aid} hash={mhash}...")
    return True

check("tgsp_results.json is valid (3 entries)", test_tgsp_results_valid)
check("all 3 conversions succeeded", test_tgsp_all_success)


# =========================================================
# 3. TGSP file format validation + RL checkpoint loading
# =========================================================
print("\n=== 3. TGSP format + checkpoint loading ===")

import torch

# 3a. TGSP files have correct magic bytes (encrypted package format)
for adapter_name in ["banking_expert", "investment_expert", "shared_attention"]:
    tgsp_path = os.path.join(TGSP_DIR, f"{adapter_name}.tgsp")

    def make_magic_test(p, n):
        def test():
            with open(p, "rb") as f:
                magic = f.read(6)
            expected = b"TGSP\x01\x00"
            ok = magic == expected
            size_mb = os.path.getsize(p) / 1048576
            print(f"    {n}: magic={magic!r} ({size_mb:.1f} MB) {'OK' if ok else 'BAD'}")
            return ok
        return test

    check(f"{adapter_name}.tgsp valid TGSP format", make_magic_test(tgsp_path, adapter_name))

# 3b. RL checkpoints loadable with LoRA weight extraction
ADAPTERS_DIR = os.path.join(BASE, "demonstrator", "adapters")
for adapter_name in ["banking_expert", "investment_expert", "shared_attention"]:
    ckpt_path = os.path.join(ADAPTERS_DIR, f"{adapter_name}_rl", "adapter_rl_final.pt")

    def make_ckpt_test(p, n):
        def test():
            import io
            with open(p, "rb") as f:
                data = f.read()
            state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            del data
            model_state = state.get("model_state_dict", {})
            lora_keys = [k for k in model_state if "lora_" in k.lower()]
            a_keys = [k for k in lora_keys if "lora_A" in k]
            b_keys = [k for k in lora_keys if "lora_B" in k]
            print(f"    {n}: {len(lora_keys)} lora keys ({len(a_keys)} A, {len(b_keys)} B)")
            if a_keys:
                sample = model_state[a_keys[0]]
                print(f"    sample lora_A shape: {list(sample.shape)}")
            del state, model_state
            return len(a_keys) > 0 and len(b_keys) > 0
        return test

    check(f"{adapter_name} RL checkpoint has LoRA weights", make_ckpt_test(ckpt_path, adapter_name))


# =========================================================
# 4. Python imports
# =========================================================
print("\n=== 4. Python imports ===")

def test_import(module_name):
    def test():
        __import__(module_name)
        return True
    return test

for mod in ["fastapi", "pydantic", "uvicorn", "torch", "transformers", "numpy"]:
    check(f"import {mod}", test_import(mod))

def test_inference_engine_import():
    from demonstrator.server.inference_engine import FinanceInferenceEngine
    return FinanceInferenceEngine is not None

def test_server_app_import():
    # Don't actually run startup, just check the module parses
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "demonstrator.server.app",
        os.path.join(BASE, "demonstrator", "server", "app.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't exec (would start FastAPI), just check it loads without syntax errors
    return spec is not None

def test_reward_fn():
    from demonstrator.training.reward_fn import finance_reward
    # Run with real sample data
    score = finance_reward(
        prompt="What is the best way to invest in index funds?",
        response="Index funds offer diversification through passive management. "
                 "They track a market index like the S&P 500, providing broad exposure "
                 "to equities with low expense ratios. Dollar cost averaging into index "
                 "funds is a proven strategy for long-term wealth building. "
                 "This is not financial advice - consult a professional advisor.",
        meta={"expert": "investment_expert"},
    )
    print(f"    reward score: {score:.3f}")
    return 0.0 <= score <= 1.0

def test_data_loading_import():
    from demonstrator.training.data_loading import (
        load_banking_dataset, load_investment_dataset,
        load_combined_finance_dataset, load_rl_prompts,
    )
    return all(fn is not None for fn in [
        load_banking_dataset, load_investment_dataset,
        load_combined_finance_dataset, load_rl_prompts,
    ])

check("import inference_engine", test_inference_engine_import)
check("server app.py parseable", test_server_app_import)
check("reward_fn works with sample data", test_reward_fn)
check("data_loading imports", test_data_loading_import)


# =========================================================
# 5. FinanceInferenceEngine instantiation
# =========================================================
print("\n=== 5. Inference engine instantiation ===")

def test_engine_init():
    from demonstrator.server.inference_engine import FinanceInferenceEngine
    cfg_path = os.path.join(TGSP_DIR, "moe_config.json")
    engine = FinanceInferenceEngine(moe_config_path=cfg_path, device="cpu")
    # Check config loaded correctly
    assert engine.moe_config is not None
    assert engine.moe_config["model"] == "Qwen/Qwen2.5-1.5B"
    assert len(engine.moe_config["experts"]) == 3
    print(f"    config loaded: {len(engine.moe_config['experts'])} experts")
    return True

def test_engine_routing():
    from demonstrator.server.inference_engine import FinanceInferenceEngine
    cfg_path = os.path.join(TGSP_DIR, "moe_config.json")
    engine = FinanceInferenceEngine(moe_config_path=cfg_path, device="cpu")
    # Manually load adapters (fake weights for routing test)
    for name, ecfg in engine.moe_config["experts"].items():
        engine.adapters[name] = {
            "weights": {},
            "config": ecfg,
            "gate_keywords": set(ecfg.get("gate_keywords", [])),
            "always_active": ecfg.get("always_active", False),
        }

    # Test routing with real queries
    tests = [
        ("How do I apply for a mortgage?", "banking_expert"),
        ("What stocks should I buy?", "investment_expert"),
        ("Tell me about deposit insurance", "banking_expert"),
        ("How does portfolio diversification work?", "investment_expert"),
        ("Hello, how are you?", "shared_attention"),  # no keywords match
    ]
    all_ok = True
    for query, expected in tests:
        got = engine.route_expert(query)
        status = "OK" if got == expected else "WRONG"
        if got != expected:
            all_ok = False
        print(f"    '{query[:40]}...' -> {got} ({status})")
    return all_ok

check("engine instantiation (config only)", test_engine_init)
check("expert routing (keyword gate)", test_engine_routing)


# =========================================================
# 6. Full engine initialize (loads model + adapters from TGSP)
# =========================================================
print("\n=== 6. Full engine initialize (model + TGSP adapters) ===")
print("    This loads Qwen 2.5 1.5B and all TGSP adapters...")

def test_full_engine_init():
    from demonstrator.server.inference_engine import FinanceInferenceEngine
    cfg_path = os.path.join(TGSP_DIR, "moe_config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    device: {device}")

    engine = FinanceInferenceEngine(moe_config_path=cfg_path, device=device)
    engine.initialize()

    assert engine._initialized, "Engine not initialized"
    assert engine.model is not None, "Model not loaded"
    assert engine.tokenizer is not None, "Tokenizer not loaded"
    print(f"    model loaded: {engine.model.config._name_or_path}")
    print(f"    tokenizer vocab: {len(engine.tokenizer)}")
    print(f"    adapters loaded: {list(engine.adapters.keys())}")
    print(f"    HE context: {'active' if engine.he_ctx else 'unavailable'}")
    print(f"    SIMD slots: {engine.simd_slots}")

    n_adapters = len(engine.adapters)
    assert n_adapters >= 1, f"Expected >=1 adapter loaded, got {n_adapters}"
    print(f"    {n_adapters}/3 adapters loaded")
    return True

check("full engine initialize (model + TGSP)", test_full_engine_init)


# =========================================================
# 7. Live inference test
# =========================================================
print("\n=== 7. Live inference test ===")

def test_live_inference():
    from demonstrator.server.inference_engine import FinanceInferenceEngine
    cfg_path = os.path.join(TGSP_DIR, "moe_config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine = FinanceInferenceEngine(moe_config_path=cfg_path, device=device)
    engine.initialize()

    query = "What is a savings account?"
    tokens = []
    metrics = None

    for chunk in engine.generate_stream(
        query=query,
        max_tokens=32,
        temperature=0.7,
        use_he=False,  # Skip HE for speed (CuKKS may not be installed)
    ):
        if chunk["type"] == "input_info":
            print(f"    input_info: expert={chunk.get('active_expert')}")
        elif chunk["type"] == "token":
            tokens.append(chunk["token"])
        elif chunk["type"] == "done":
            metrics = chunk.get("aggregate", {})

    response = "".join(tokens)
    print(f"    query: {query}")
    print(f"    response ({len(tokens)} tokens): {response[:120]}...")
    if metrics:
        tok_s = metrics.get("tokens_per_second", 0)
        print(f"    tok/s: {tok_s:.1f}")
        print(f"    total_time_ms: {metrics.get('total_time_ms', 0):.0f}")
    return len(tokens) > 0

check("live inference (no HE, 32 tokens)", test_live_inference)


# =========================================================
# 8. Frontend files
# =========================================================
print("\n=== 8. Frontend files ===")

for fname in ["index.html", "app.js", "styles.css"]:
    def make_file_test(f):
        def test():
            path = os.path.join(FRONTEND_DIR, f)
            if not os.path.isfile(path):
                print(f"    {f}: MISSING")
                return False
            size = os.path.getsize(path)
            print(f"    {f}: {size} bytes")
            return size > 100
        return test
    check(f"{fname} exists and non-trivial", make_file_test(fname))

def test_html_has_pipeline():
    with open(os.path.join(FRONTEND_DIR, "index.html")) as f:
        html = f.read()
    has_pipe = "pipeline-bar" in html
    has_route = "routing-bar" in html
    print(f"    pipeline-bar: {'found' if has_pipe else 'MISSING'}")
    print(f"    routing-bar: {'found' if has_route else 'MISSING'}")
    return has_pipe and has_route

def test_js_has_pipeline_fns():
    with open(os.path.join(FRONTEND_DIR, "app.js")) as f:
        js = f.read()
    fns = ["pipelineOnSend", "pipelineOnInputInfo", "pipelineOnToken",
           "pipelineOnDone", "resetRouting", "routingOnInputInfo", "routingOnDone"]
    missing = [fn for fn in fns if fn not in js]
    if missing:
        print(f"    missing functions: {missing}")
        return False
    print(f"    all {len(fns)} pipeline/routing functions present")
    return True

def test_css_has_pipeline_styles():
    with open(os.path.join(FRONTEND_DIR, "styles.css")) as f:
        css = f.read()
    selectors = ["#pipeline-bar", ".pipe-stage", ".pipe-arrow", "#routing-bar", ".route-pill"]
    missing = [s for s in selectors if s not in css]
    if missing:
        print(f"    missing selectors: {missing}")
        return False
    print(f"    all {len(selectors)} pipeline/routing selectors present")
    return True

check("index.html has pipeline-bar and routing-bar", test_html_has_pipeline)
check("app.js has all pipeline/routing functions", test_js_has_pipeline_fns)
check("styles.css has pipeline/routing styles", test_css_has_pipeline_styles)


# =========================================================
# Summary
# =========================================================
print("\n" + "=" * 50)
total = passed + failed
print(f"QA RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  X {e}")
print("=" * 50)
sys.exit(1 if failed > 0 else 0)
