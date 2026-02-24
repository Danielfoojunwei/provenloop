#!/usr/bin/env python3
"""
Split-mode benchmark: WebSocket vs HTTP vs Selftest
Runs in WSL with torch GPU for LM head projection (simulates optimized client).
"""
import asyncio
import base64
import json
import time

import numpy as np
import requests
import torch
import websockets

# ── Config ────────────────────────────────────────────────────────────────
SERVER = "127.0.0.1:8095"
QUERY = "What is a savings account"
MAX_TOKENS = 32
HD = 1536  # hidden_dim for Qwen2.5-1.5B

PROMPT_TEMPLATE = (
    "### System:\nYou are a helpful financial assistant.\n\n"
    "### Instruction:\n{query}\n\n### Response:\n"
)


def load_weights(weights_dir: str):
    """Load embed_tokens and lm_head as torch tensors on GPU."""
    import pathlib
    wdir = pathlib.Path(weights_dir)

    # embed_tokens.bin — float16, shape (VS, HD)
    embed_f16 = np.fromfile(str(wdir / "embed_tokens.bin"), dtype=np.float16)
    VS = embed_f16.shape[0] // HD
    embed_f16 = embed_f16.reshape(VS, HD)

    # Load as torch on GPU
    embed_gpu = torch.tensor(embed_f16, dtype=torch.float16, device="cuda")

    # LM head uses same embed_tokens weights (weight tying in Qwen2.5)
    lm_head_gpu = embed_gpu.float()  # (VS, HD) float32 on GPU

    print(f"Loaded weights: VS={VS}, HD={HD}, embed={embed_gpu.shape}, lm_head={lm_head_gpu.shape}")
    return embed_gpu, lm_head_gpu, VS


def embed_tokens(token_ids, embed_gpu):
    """Embed token IDs using GPU. Returns (seq_len, HD) float32 numpy."""
    indices = torch.tensor(token_ids, dtype=torch.long, device="cuda")
    emb = embed_gpu[indices].float()  # (seq_len, HD) float32 on GPU
    return emb.cpu().numpy()


def lm_head_project(hidden_f32_np, lm_head_gpu):
    """Project hidden state through LM head on GPU. Returns logits numpy."""
    h_gpu = torch.tensor(hidden_f32_np, dtype=torch.float32, device="cuda")
    logits = lm_head_gpu @ h_gpu  # (VS,) float32 on GPU
    return logits.cpu().numpy()


def tokenize_server(query: str):
    """Tokenize via server HTTP."""
    text = PROMPT_TEMPLATE.format(query=query)
    r = requests.post(
        f"http://{SERVER}/api/v1/split/tokenize",
        json={"text": text}
    )
    return r.json()["token_ids"]


def route_expert(query: str):
    """Simple keyword-based routing (matches server logic)."""
    q = query.lower()
    banking_kw = ["bank", "deposit", "loan", "mortgage", "credit", "savings", "checking", "interest rate", "refinance"]
    invest_kw = ["invest", "portfolio", "stock", "bond", "etf", "dividend", "market", "allocation", "risk"]
    for kw in banking_kw:
        if kw in q:
            return "banking_expert"
    for kw in invest_kw:
        if kw in q:
            return "investment_expert"
    return "shared_attention"


# ── HTTP Split Client ─────────────────────────────────────────────────────

def bench_http_split(token_ids, embed_gpu, lm_head_gpu, query):
    """Benchmark split mode using HTTP POST per token."""
    expert = route_expert(query)
    session_id = f"bench_http_{int(time.time())}"
    generated = []
    cur_ids = list(token_ids)

    print(f"\n{'='*60}")
    print(f"HTTP Split Benchmark ({MAX_TOKENS} tokens, expert={expert})")
    print(f"{'='*60}")

    t_start = time.perf_counter()
    step_times = []

    for step in range(MAX_TOKENS):
        t_step = time.perf_counter()
        incremental = step > 0

        # 1. Embed (GPU)
        if incremental:
            emb_np = embed_tokens([cur_ids[-1]], embed_gpu)
        else:
            emb_np = embed_tokens(cur_ids, embed_gpu)

        # 2. HTTP POST to server
        h_b64 = base64.b64encode(emb_np.astype(np.float32).tobytes()).decode()
        r = requests.post(
            f"http://{SERVER}/api/v1/split/forward",
            json={
                "hidden_states_b64": h_b64,
                "seq_len": emb_np.shape[0],
                "hidden_dim": HD,
                "expert_name": expert,
                "use_he": True,
                "session_id": session_id,
                "incremental": incremental,
            }
        )
        resp = r.json()

        # 3. Decode pre-activations
        pre_b64 = resp["pre_activations_b64"]
        pre_np = np.frombuffer(base64.b64decode(pre_b64), dtype=np.float32)

        # Extract last hidden state
        if incremental:
            last_h = pre_np[:HD]
        else:
            last_h = pre_np.reshape(-1, HD)[-1]

        # 4. LM head projection (GPU)
        logits = lm_head_project(last_h, lm_head_gpu)

        # 5. Greedy sample
        next_id = int(np.argmax(logits))

        step_ms = (time.perf_counter() - t_step) * 1000
        step_times.append(step_ms)

        # EOS check
        if next_id in (151643, 151645):  # eos_token, im_end
            break

        cur_ids.append(next_id)
        generated.append(next_id)

        if step < 3:
            print(f"  step {step}: {step_ms:.0f}ms (server: {resp.get('total_ms', 0):.0f}ms)")

    t_total = (time.perf_counter() - t_start) * 1000
    n_tok = len(generated)
    tok_s = n_tok / (t_total / 1000) if t_total > 0 else 0

    print(f"\n  Tokens: {n_tok}")
    print(f"  Total:  {t_total:.0f}ms")
    print(f"  tok/s:  {tok_s:.2f}")
    print(f"  Avg/tok: {t_total/max(n_tok,1):.0f}ms")
    if step_times:
        print(f"  Median step: {sorted(step_times)[len(step_times)//2]:.0f}ms")

    return tok_s, n_tok, t_total


# ── WebSocket Split Client ────────────────────────────────────────────────

async def bench_ws_split(token_ids, embed_gpu, lm_head_gpu, query):
    """Benchmark split mode using persistent WebSocket."""
    expert = route_expert(query)
    session_id = f"bench_ws_{int(time.time())}"
    generated = []
    cur_ids = list(token_ids)

    print(f"\n{'='*60}")
    print(f"WebSocket Split Benchmark ({MAX_TOKENS} tokens, expert={expert})")
    print(f"{'='*60}")

    uri = f"ws://{SERVER}/api/v1/split/stream"
    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        # Ping to verify connection
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        assert pong["type"] == "pong", f"Expected pong, got {pong}"
        print("  WS connected, ping OK")

        t_start = time.perf_counter()
        step_times = []

        for step in range(MAX_TOKENS):
            t_step = time.perf_counter()
            incremental = step > 0

            # 1. Embed (GPU)
            if incremental:
                emb_np = embed_tokens([cur_ids[-1]], embed_gpu)
            else:
                emb_np = embed_tokens(cur_ids, embed_gpu)

            # 2. WebSocket forward
            h_b64 = base64.b64encode(emb_np.astype(np.float32).tobytes()).decode()
            await ws.send(json.dumps({
                "type": "forward",
                "hidden_states_b64": h_b64,
                "seq_len": emb_np.shape[0],
                "hidden_dim": HD,
                "expert_name": expert,
                "use_he": True,
                "session_id": session_id,
                "incremental": incremental,
            }))
            resp = json.loads(await ws.recv())

            if resp.get("type") == "error":
                print(f"  ERROR at step {step}: {resp.get('message')}")
                break

            # 3. Decode pre-activations
            pre_b64 = resp["pre_activations_b64"]
            pre_np = np.frombuffer(base64.b64decode(pre_b64), dtype=np.float32)

            # Extract last hidden state
            if incremental:
                last_h = pre_np[:HD]
            else:
                last_h = pre_np.reshape(-1, HD)[-1]

            # 4. LM head projection (GPU)
            logits = lm_head_project(last_h, lm_head_gpu)

            # 5. Greedy sample
            next_id = int(np.argmax(logits))

            step_ms = (time.perf_counter() - t_step) * 1000
            step_times.append(step_ms)

            # EOS check
            if next_id in (151643, 151645):
                break

            cur_ids.append(next_id)
            generated.append(next_id)

            if step < 3:
                print(f"  step {step}: {step_ms:.0f}ms (server: {resp.get('total_ms', 0):.0f}ms)")

    t_total = (time.perf_counter() - t_start) * 1000
    n_tok = len(generated)
    tok_s = n_tok / (t_total / 1000) if t_total > 0 else 0

    print(f"\n  Tokens: {n_tok}")
    print(f"  Total:  {t_total:.0f}ms")
    print(f"  tok/s:  {tok_s:.2f}")
    print(f"  Avg/tok: {t_total/max(n_tok,1):.0f}ms")
    if step_times:
        print(f"  Median step: {sorted(step_times)[len(step_times)//2]:.0f}ms")

    return tok_s, n_tok, t_total


# ── Selftest baseline (numpy LM head, no network) ────────────────────────

def bench_selftest():
    """Run server selftest and time it externally."""
    print(f"\n{'='*60}")
    print(f"Server Selftest Baseline (numpy LM head, no network)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    r = requests.get(
        f"http://{SERVER}/api/v1/split/selftest",
        params={"query": QUERY}
    )
    t_total = (time.perf_counter() - t0) * 1000
    d = r.json()
    n_tok = d.get("total_tokens", 0)
    tok_s = n_tok / (t_total / 1000) if t_total > 0 else 0

    print(f"  Tokens: {n_tok}")
    print(f"  Total:  {t_total:.0f}ms")
    print(f"  tok/s:  {tok_s:.2f}")
    print(f"  Verdict: {d.get('verdict', 'N/A')}")
    print(f"  DP sigma: {d.get('dp_sigma', 0)}")

    return tok_s, n_tok, t_total


# ── Chat/compare baseline (server-side generate_stream) ──────────────────

def bench_chat_compare():
    """Run chat/compare for WebSocket-mode baseline."""
    print(f"\n{'='*60}")
    print(f"WebSocket Mode Baseline (server-side, torch GPU LM head)")
    print(f"{'='*60}")

    r = requests.post(
        f"http://{SERVER}/api/v1/chat/compare",
        json={"query": QUERY, "max_tokens": MAX_TOKENS}
    )
    d = r.json()
    base = d.get("base", {})
    adapted = d.get("adapted", {})

    print(f"  Base (no HE):    {base.get('tok_s', 0):.1f} tok/s ({base.get('time_ms', 0):.0f}ms, {base.get('tokens', 0)} tokens)")
    print(f"  Adapted (HE):    {adapted.get('tok_s', 0):.1f} tok/s ({adapted.get('time_ms', 0):.0f}ms, {adapted.get('tokens', 0)} tokens)")

    return adapted.get("tok_s", 0)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    weights_dir = "/mnt/c/Users/lover/Downloads/TenSafe_Project/demonstrator/frontend/weights"

    print("Loading weights to GPU...")
    embed_gpu, lm_head_gpu, VS = load_weights(weights_dir)

    print("\nTokenizing query...")
    token_ids = tokenize_server(QUERY)
    print(f"  Input tokens: {len(token_ids)}")
    print(f"  Expert: {route_expert(QUERY)}")

    # 1. Server selftest baseline (numpy LM head)
    selftest_toks, _, _ = bench_selftest()

    # 2. Chat/compare baseline (WebSocket mode, fully server-side)
    ws_mode_toks = bench_chat_compare()

    # 3. HTTP split with GPU LM head
    http_toks, _, _ = bench_http_split(token_ids, embed_gpu, lm_head_gpu, QUERY)

    # 4. WebSocket split with GPU LM head
    ws_toks, _, _ = asyncio.run(bench_ws_split(token_ids, embed_gpu, lm_head_gpu, QUERY))

    # ── Summary ───────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"BENCHMARK SUMMARY — Split Mode Optimization")
    print(f"{sep}")
    print(f"  Server: CuKKS GPU, poly_n=16384, batch decrypt")
    print(f"  Client: WSL Python, torch GPU LM head")
    print(f"  Query:  '{QUERY}'")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"{sep}")
    print(f"")
    print(f"  {'Mode':<40} {'tok/s':>8}  {'vs baseline':>12}")
    print(f"  {'-'*40} {'-'*8}  {'-'*12}")
    print(f"  {'Selftest (numpy CPU LM head)':<40} {selftest_toks:>8.2f}  {'baseline':>12}")
    print(f"  {'HTTP + GPU LM head':<40} {http_toks:>8.2f}  {f'+{(http_toks/max(selftest_toks,0.01)-1)*100:.0f}%':>12}")
    print(f"  {'WebSocket + GPU LM head':<40} {ws_toks:>8.2f}  {f'+{(ws_toks/max(selftest_toks,0.01)-1)*100:.0f}%':>12}")
    print(f"  {'WS Mode (fully server-side, reference)':<40} {ws_mode_toks:>8.1f}  {'(server ref)':>12}")
    print(f"{sep}")


if __name__ == "__main__":
    main()
