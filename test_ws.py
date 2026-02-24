import asyncio, websockets, json

async def test():
    uri = "ws://localhost:8000/api/v1/chat/stream"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "query": "What is a savings account?",
            "max_tokens": 5,
            "use_he": True,
        }))
        async for msg in ws:
            data = json.loads(msg)
            t = data["type"]
            if t == "token":
                print(data["token"], end="", flush=True)
            elif t == "done":
                agg = data["aggregate"]
                toks = agg["total_tokens"]
                tps = agg["tokens_per_second"]
                print()
                print("DONE: %d tokens, %.1f tok/s" % (toks, tps))
                break
            elif t == "input_info":
                expert = data["active_expert"]
                enc = data["encrypted"]
                print("[Expert: %s, HE: %s]" % (expert, enc))
            elif t == "error":
                print("ERROR: " + data.get("message", "unknown"))
                break

asyncio.run(test())
