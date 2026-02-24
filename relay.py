"""TCP relay: 0.0.0.0:9090 -> 127.0.0.1:8090 (exposes WSL to LAN).

Handles large file transfers (446 MB weight download) with big buffers.
Logs connection activity so we can debug phone issues.
"""
import asyncio
import sys
import time

LOCAL = ("127.0.0.1", 8095)
LISTEN = ("0.0.0.0", 9095)
BUF = 1024 * 1024  # 1 MB chunks for large weight downloads

_conn_id = 0

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def pipe(label, r, w, conn_tag):
    total = 0
    try:
        while True:
            data = await r.read(BUF)
            if not data:
                break
            w.write(data)
            await w.drain()
            total += len(data)
    except (ConnectionError, BrokenPipeError, asyncio.CancelledError):
        pass
    except Exception as e:
        log(f"  {conn_tag} {label} error: {e}")
    finally:
        try:
            w.close()
            await w.wait_closed()
        except Exception:
            pass
    if total > 100000:  # Only log large transfers
        log(f"  {conn_tag} {label} done: {total / 1048576:.1f} MB")


async def handle(cr, cw):
    global _conn_id
    _conn_id += 1
    tag = f"[C{_conn_id}]"

    peer = cw.get_extra_info("peername", ("?", 0))
    log(f"{tag} New connection from {peer[0]}:{peer[1]}")

    try:
        sr, sw = await asyncio.open_connection(*LOCAL)
    except Exception as e:
        log(f"{tag} Cannot reach backend {LOCAL[0]}:{LOCAL[1]}: {e}")
        try:
            cw.close()
        except Exception:
            pass
        return

    await asyncio.gather(
        pipe("client→server", cr, sw, tag),
        pipe("server→client", sr, cw, tag),
    )
    log(f"{tag} Connection closed")


async def main():
    srv = await asyncio.start_server(handle, *LISTEN)
    log(f"Relay listening on {LISTEN[0]}:{LISTEN[1]} -> {LOCAL[0]}:{LOCAL[1]}")
    log(f"Phone URL: http://<LAN-IP>:{LISTEN[1]}")
    async with srv:
        await srv.serve_forever()

asyncio.run(main())
