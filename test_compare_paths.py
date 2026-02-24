"""Compare server generate_stream vs split path embeddings."""
import numpy as np
import sys, os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from demonstrator.server.inference_engine import FinanceInferenceEngine
import torch

engine = FinanceInferenceEngine(
    moe_config_path="demonstrator/adapters/tgsp/moe_config.json",
    device="cuda",
)
engine.initialize()

# 1. Check: do embed_tokens.bin weights match model's embed_tokens?
HD = 1536
w_file = np.fromfile("demonstrator/frontend/weights/embed_tokens.bin", dtype=np.float16).reshape(-1, HD)
w_model = engine.model.model.embed_tokens.weight.detach().cpu().numpy()  # float16 or float32

print(f"embed_tokens.bin shape: {w_file.shape}")
print(f"model embed_tokens shape: {w_model.shape}")

# Compare first 10 tokens
for tid in [0, 1, 100, 1000, 10000, 50000, 100000, 151644]:
    file_vec = w_file[tid].astype(np.float32)
    model_vec = w_model[tid].astype(np.float32)
    diff = np.abs(file_vec - model_vec).max()
    dot = np.dot(file_vec, model_vec) / (np.linalg.norm(file_vec) * np.linalg.norm(model_vec) + 1e-10)
    print(f"  token {tid:6d}: max_diff={diff:.6f}  cosine={dot:.6f}")

# 2. Check: does model's lm_head == embed_tokens (tied)?
lm_head = engine.model.lm_head.weight.detach().cpu().numpy()
print(f"\nlm_head shape: {lm_head.shape}")
print(f"lm_head == embed_tokens? {np.array_equal(w_model, lm_head)}")
if not np.array_equal(w_model, lm_head):
    diff = np.abs(w_model.astype(np.float32) - lm_head.astype(np.float32)).max()
    print(f"  max_diff: {diff}")

# 3. Run full server-side generation for "hello" with ChatML
print("\n--- Server generate_stream (full path, no split) ---")
tokens = []
for chunk in engine.generate_stream(
    "hello", max_tokens=20, temperature=0.01, top_k=1,
    use_he=True, session_id="compare_full",
):
    if chunk["type"] == "token":
        tokens.append(chunk["token"])
        print(f"  {repr(chunk['token'])}", end="", flush=True)
print(f"\nFull response: {''.join(tokens)}")

# 4. Now test split with model's OWN embedding
print("\n--- Split path using MODEL's embedding (not embed_tokens.bin) ---")
tok = engine.tokenizer
prompt = (
    "<|im_start|>system\n"
    "You are a helpful financial assistant. Answer clearly and concisely in English.<|im_end|>\n"
    "<|im_start|>user\nhello<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids = tok.encode(prompt)
# Use MODEL's embeddings
with torch.no_grad():
    model_emb = engine.model.model.embed_tokens(
        torch.tensor([ids], device="cuda")
    )[0].cpu().numpy().astype(np.float32)

print(f"Model embedding shape: {model_emb.shape}")

# Split forward with model embeddings
result = engine.split_forward(
    model_emb, "shared_attention", use_he=True,
    session_id="compare_split_model", incremental=False,
)
pa = result["pre_activations"].reshape(-1, HD)
last_h = pa[-1]

# Project with model's lm_head
with torch.no_grad():
    lh_tensor = torch.tensor(last_h, dtype=torch.float16, device="cuda").unsqueeze(0).unsqueeze(0)
    logits_model = engine.model.lm_head(lh_tensor).squeeze().cpu().numpy()

top5 = np.argsort(logits_model)[-5:][::-1]
print("Top-5 (model lm_head):")
for tid in top5:
    print(f"  id={tid:6d} score={logits_model[tid]:.2f} -> {repr(tok.decode([tid]))}")

# 5. Now project with embed_tokens.bin (what phone client does)
logits_file = w_file.astype(np.float32) @ last_h
top5f = np.argsort(logits_file)[-5:][::-1]
print("\nTop-5 (embed_tokens.bin matmul):")
for tid in top5f:
    print(f"  id={tid:6d} score={logits_file[tid]:.2f} -> {repr(tok.decode([tid]))}")
