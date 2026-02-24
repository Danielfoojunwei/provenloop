"""Compare JS BPE tokenizer vs HuggingFace tokenizer output."""
import json, re, os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from transformers import AutoTokenizer
hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# Load JS tokenizer data
with open("demonstrator/frontend/weights/tokenizer.json") as f:
    tj = json.load(f)

m = tj["model"]
vocab = m["vocab"]
merges = m["merges"]
added = {a["content"]: a["id"] for a in (tj.get("added_tokens") or [])}

# Replicate JS BPE tokenizer logic
BYTE_TO_CHAR = {}
n = 0
for b in range(256):
    if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
        BYTE_TO_CHAR[b] = chr(b)
    else:
        BYTE_TO_CHAR[b] = chr(256 + n)
        n += 1

def text_to_byte_chars(text):
    return "".join(BYTE_TO_CHAR[b] for b in text.encode("utf-8"))

# Build ranks
ranks = {}
for i, merge in enumerate(merges):
    ranks[merge] = i

def bpe(word):
    sym = list(word)
    if len(sym) <= 1:
        return sym
    while True:
        best_r = float("inf")
        best_p = None
        for i in range(len(sym) - 1):
            pair = sym[i] + " " + sym[i + 1]
            r = ranks.get(pair)
            if r is not None and r < best_r:
                best_r = r
                best_p = (sym[i], sym[i + 1])
        if best_p is None:
            break
        ns = []
        i = 0
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == best_p[0] and sym[i + 1] == best_p[1]:
                ns.append(best_p[0] + best_p[1])
                i += 2
            else:
                ns.append(sym[i])
                i += 1
        sym = ns
        if len(sym) == 1:
            break
    return sym

# Pre-tokenization regex (from Qwen tokenizer)
# Check what the JS uses
PRE_RE_STR = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Python equivalent (re doesn't support \p{L}, use regex module)
try:
    import regex
    PRE_RE = regex.compile(PRE_RE_STR)
    has_regex = True
except ImportError:
    has_regex = False
    # Fallback: basic split
    PRE_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+""")

def js_encode(text):
    ids = []
    rem = text
    while len(rem) > 0:
        found = False
        for tok_str, tok_id in added.items():
            if rem.startswith(tok_str):
                ids.append(tok_id)
                rem = rem[len(tok_str):]
                found = True
                break
        if found:
            continue
        # Find next special token
        nxt = len(rem)
        for tok_str in added:
            p = rem.find(tok_str)
            if p > 0 and p < nxt:
                nxt = p
        chunk = rem[:nxt]
        rem = rem[nxt:]
        words = PRE_RE.findall(chunk)
        for w in words:
            bw = text_to_byte_chars(w)
            for t in bpe(bw):
                if t in vocab:
                    ids.append(vocab[t])
        # else: skip unknown token
    return ids

# Test strings
tests = [
    "hello",
    "What is a savings account?",
    "Explain what is a DCF in finance terms",
    "### System:\nYou are a helpful financial assistant.\n\n### Instruction:\nhello\n\n### Response:\n",
    "### System:\nYou are a helpful financial assistant.\n\n### Instruction:\nExplain what is a DCF in finance terms\n\n### Response:\n",
]

print(f"Using regex module: {has_regex}")
print()

for text in tests:
    hf_ids = hf_tok.encode(text)
    js_ids = js_encode(text)
    match = hf_ids == js_ids
    print(f"Text: {repr(text[:80])}")
    print(f"  HF tokens: {len(hf_ids)} -> {hf_ids[:15]}{'...' if len(hf_ids) > 15 else ''}")
    print(f"  JS tokens: {len(js_ids)} -> {js_ids[:15]}{'...' if len(js_ids) > 15 else ''}")
    print(f"  MATCH: {match}")
    if not match:
        # Find first difference
        for i in range(min(len(hf_ids), len(js_ids))):
            if hf_ids[i] != js_ids[i]:
                hf_t = hf_tok.decode([hf_ids[i]])
                js_t = hf_tok.decode([js_ids[i]]) if js_ids[i] < hf_tok.vocab_size else "???"
                print(f"  FIRST DIFF at pos {i}: HF={hf_ids[i]}({repr(hf_t)}) vs JS={js_ids[i]}({repr(js_t)})")
                break
        if len(hf_ids) != len(js_ids):
            print(f"  LENGTH DIFF: HF={len(hf_ids)} vs JS={len(js_ids)}")
    print()
