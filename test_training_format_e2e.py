"""E2E: verify training-format prompt produces good output through split path."""
import numpy as np, base64, json, urllib.request

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

HD = 1536
w = np.fromfile("demonstrator/frontend/weights/embed_tokens.bin", dtype=np.float16).reshape(-1, HD)

def split_generate(query, max_steps=30):
    # Training format (matches LoRA fine-tuning)
    prompt = f"### System:\nYou are a helpful financial assistant.\n\n### Instruction:\n{query}\n\n### Response:\n"
    ids = tok.encode(prompt)

    cur_ids = list(ids)
    generated = []
    session = f"test_{hash(query) % 10000}"

    for step in range(max_steps):
        if step == 0:
            emb = np.zeros((len(ids), HD), dtype=np.float32)
            for i, tid in enumerate(ids):
                emb[i] = w[tid].astype(np.float32)
            seq_len = len(ids)
            incremental = False
        else:
            emb = w[cur_ids[-1]].astype(np.float32).reshape(1, HD)
            seq_len = 1
            incremental = True

        payload = json.dumps({
            "hidden_states_b64": base64.b64encode(emb.tobytes()).decode(),
            "seq_len": seq_len,
            "hidden_dim": HD,
            "expert_name": "shared_attention",
            "use_he": True,
            "session_id": session,
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
        last_h = pa[-1]
        logits = w.astype(np.float32) @ last_h

        # Repetition penalty
        seen = set(cur_ids[-64:])
        for tid in seen:
            if logits[tid] > 0:
                logits[tid] /= 1.3
            else:
                logits[tid] *= 1.3

        next_id = int(np.argmax(logits))  # greedy

        # Stop conditions
        if next_id in (151643, 151645):  # EOS or im_end
            break
        t = tok.decode([next_id])
        generated.append(t)
        cur_ids.append(next_id)

        # Stop on section boundary
        if len(generated) > 2:
            tail = "".join(generated[-4:])
            if "\n###" in tail:
                break

    return "".join(generated).strip()

# Test queries
queries = [
    "hello",
    "What is a savings account?",
    "How does a mortgage work?",
    "What is the best way to invest money?",
]

for q in queries:
    result = split_generate(q)
    print(f"\nQ: {q}")
    print(f"A: {result[:200]}")
    # Quality checks
    has_chinese = any(ord(c) > 0x4E00 and ord(c) < 0x9FFF for c in result)
    is_repetitive = len(set(result.split())) < len(result.split()) * 0.3 if result.split() else False
    print(f"   Chinese: {'YES (BAD)' if has_chinese else 'No'}")
    print(f"   Repetitive: {'YES (BAD)' if is_repetitive else 'No'}")
    print(f"   Length: {len(result)} chars")
