"""E2E test: simulate phone client with ChatML template."""
import numpy as np, base64, json, urllib.request

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

prompt = (
    "<|im_start|>system\n"
    "You are a helpful financial assistant. Answer clearly and concisely in English.<|im_end|>\n"
    "<|im_start|>user\nhello<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids = tok.encode(prompt)
print(f"Prompt tokens: {len(ids)}")

# Load embed weights (same file the phone downloads)
HD = 1536
w = np.fromfile("demonstrator/frontend/weights/embed_tokens.bin", dtype=np.float16).reshape(-1, HD)
print(f"Weights: {w.shape}")

# Embed all tokens (what the phone client does)
emb = np.zeros((len(ids), HD), dtype=np.float32)
for i, tid in enumerate(ids):
    emb[i] = w[tid].astype(np.float32)

# Send to server split/forward
h_b64 = base64.b64encode(emb.tobytes()).decode()
payload = json.dumps({
    "hidden_states_b64": h_b64,
    "seq_len": len(ids),
    "hidden_dim": HD,
    "expert_name": "shared_attention",
    "use_he": True,
    "session_id": "chatml_test_1",
    "incremental": False,
}).encode()

req = urllib.request.Request(
    "http://localhost:8090/api/v1/split/forward",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read())
print(f"HE active: {data.get('he_active')}")
print(f"Layers: {data.get('layers_computed')}")

# LM head projection (what phone does)
pre_act = np.frombuffer(
    base64.b64decode(data["pre_activations_b64"]), dtype=np.float32
).reshape(-1, HD)
last_h = pre_act[-1]
logits = w.astype(np.float32) @ last_h

# Greedy top-5
top5 = np.argsort(logits)[-5:][::-1]
print("\nTop-5 next tokens after ChatML 'hello':")
for tid in top5:
    t = tok.decode([tid])
    print(f"  id={tid:6d}  score={logits[tid]:8.2f}  -> {repr(t)}")

# Generate 20 tokens autoregressively
print("\n--- Autoregressive generation (20 tokens) ---")
cur_ids = list(ids)
generated = []
for step in range(20):
    if step == 0:
        send_emb = emb
        seq_len = len(ids)
        incremental = False
    else:
        new_emb = w[cur_ids[-1]].astype(np.float32).reshape(1, HD)
        send_emb = new_emb
        seq_len = 1
        incremental = True

    h_b64 = base64.b64encode(send_emb.tobytes()).decode()
    payload = json.dumps({
        "hidden_states_b64": h_b64,
        "seq_len": seq_len,
        "hidden_dim": HD,
        "expert_name": "shared_attention",
        "use_he": True,
        "session_id": "chatml_test_1",
        "incremental": incremental,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:8090/api/v1/split/forward",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    pa = np.frombuffer(
        base64.b64decode(data["pre_activations_b64"]), dtype=np.float32
    ).reshape(-1, HD)
    lh = pa[-1]
    lg = w.astype(np.float32) @ lh

    # Greedy
    next_id = int(np.argmax(lg))

    # Stop on EOS or im_end
    if next_id in (151643, 151645):
        print(f"[EOS at step {step}]")
        break

    cur_ids.append(next_id)
    t = tok.decode([next_id])
    generated.append(t)
    print(f"  step {step:2d}: id={next_id:6d} -> {repr(t)}")

print(f"\nFull response: {''.join(generated)}")
