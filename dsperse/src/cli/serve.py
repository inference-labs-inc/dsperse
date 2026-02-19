import asyncio
import logging
import signal

from dsperse.src.server.handlers import ServerContext
from dsperse.src.server.ipc import start_server

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "/tmp/dsperse.sock"


def setup_parser(subparsers):
    parser = subparsers.add_parser("serve", help="Start the IPC server for subnet-2 integration")
    parser.set_defaults(command="serve")
    parser.add_argument(
        "--socket-path",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--circuit-cache-dir",
        default=None,
        help="Circuit cache directory (default: ~/.bittensor/subnet-2/circuit_cache)",
    )
    return parser


def serve(args):
    socket_path = getattr(args, "socket_path", DEFAULT_SOCKET_PATH)
    cache_dir = getattr(args, "circuit_cache_dir", None)
    ctx = ServerContext(circuit_cache_dir=cache_dir)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)

    logger.info(f"Starting IPC server on {socket_path}")
    try:
        loop.run_until_complete(start_server(socket_path, ctx.dispatch))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
