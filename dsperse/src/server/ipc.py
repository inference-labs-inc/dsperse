import asyncio
import json
import logging
import os
import struct
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

MAX_PAYLOAD = 64 * 1024 * 1024


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    handler: Callable[[dict], Awaitable[dict]],
):
    response = None
    try:
        len_buf = await reader.readexactly(4)
        (payload_len,) = struct.unpack(">I", len_buf)
        if payload_len > MAX_PAYLOAD:
            response = {"error": f"payload length {payload_len} exceeds {MAX_PAYLOAD}"}
        else:
            raw = await reader.readexactly(payload_len)
            request = json.loads(raw)
            response = await handler(request)
    except asyncio.IncompleteReadError:
        pass
    except json.JSONDecodeError as e:
        response = {"error": f"invalid JSON: {e}"}
    except Exception as e:
        logger.exception("IPC handler error")
        response = {"error": str(e)}
    finally:
        try:
            if response is not None:
                resp_bytes = json.dumps(response).encode()
                writer.write(struct.pack(">I", len(resp_bytes)))
                writer.write(resp_bytes)
                await writer.drain()
        except Exception:
            logger.debug("failed to send IPC response", exc_info=True)
        writer.close()
        await writer.wait_closed()


async def start_server(
    socket_path: str,
    handler: Callable[[dict], Awaitable[dict]],
):
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    async def on_connect(reader, writer):
        await _handle_connection(reader, writer, handler)

    old_umask = os.umask(0o177)
    try:
        server = await asyncio.start_unix_server(on_connect, path=socket_path)
    finally:
        os.umask(old_umask)
    logger.info("IPC server listening on %s", socket_path)
    try:
        async with server:
            await server.serve_forever()
    finally:
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
