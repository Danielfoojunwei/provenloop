import asyncio, websockets, json

async def test():
    uri = "ws://localhost:8000/api/v1/chat/stream"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "query": "What is a bond?",
            "max_tokens": 10,
            "use_he": True,
        }))
        async for msg in ws:
            data = json.loads(msg)
            t = data["type"]
            if t == "token":
                m = data.get("metrics", {})
                agg = data["aggregate"]
                print(
                    f"  tok={data['token']!r:10s}"
                    f"  total={agg['total_time_ms']:.0f}ms"
                    f"  enc={m.get('encrypt_ms',0):.0f}ms"
                    f"  comp={m.get('compute_ms',0):.0f}ms"
                    f"  dec={m.get('decrypt_ms',0):.0f}ms"
                    f"  net={m.get('network_ms',0):.0f}ms"
                    f"  tok_lat={m.get('latency_ms',0):.0f}ms"
                )
            elif t == "done":
                agg = data["aggregate"]
                print(f"\nDONE: {agg['total_tokens']} tokens, {agg['tokens_per_second']:.2f} tok/s")
                print(f"  total_time={agg['total_time_ms']:.0f}ms")
                print(f"  enc_total={agg['total_encrypt_ms']:.0f}ms")
                print(f"  comp_total={agg['total_compute_ms']:.0f}ms")
                print(f"  dec_total={agg['total_decrypt_ms']:.0f}ms")
                print(f"  net_total={agg['total_network_ms']:.0f}ms")
                break
            elif t == "input_info":
                print(f"[Expert: {data['active_expert']}, HE: {data['encrypted']}, enc_t={data['encrypt_time_ms']}ms]")
            elif t == "error":
                print(f"ERROR: {data.get('message', 'unknown')}")
                break

asyncio.run(test())
