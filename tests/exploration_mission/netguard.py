"""The network is blocked in every File 21 test (spec hard rule 6, step 1).

    import netguard; netguard.block()

After block(), socket.socket.connect / connect_ex and socket.create_connection
raise NetworkBlocked for anything that is not loopback or a Unix socket, so a
test that passes has proven the code path never reached the internet. It never
touches the Pi's own network settings (CLAUDE.md PART 1) -- it patches this
process's socket module and nothing else.
"""

from __future__ import annotations

import ipaddress
import socket


class NetworkBlocked(OSError):
    pass


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection
attempts: list[str] = []


def _allowed(sock_family, address) -> bool:
    if sock_family == getattr(socket, "AF_UNIX", object()):
        return True
    host = address[0] if isinstance(address, tuple) else address
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False                    # a name that would need DNS: blocked


def _guard(sock, address) -> None:
    if not _allowed(sock.family, address):
        attempts.append(repr(address))
        raise NetworkBlocked(f"network blocked in File 21 tests: {address!r}")


def _connect(self, address):
    _guard(self, address)
    return _real_connect(self, address)


def _connect_ex(self, address):
    _guard(self, address)
    return _real_connect_ex(self, address)


def _create_connection(address, *args, **kwargs):
    host = address[0]
    fam = socket.AF_INET6 if ":" in str(host) else socket.AF_INET
    if not _allowed(fam, address):
        attempts.append(repr(address))
        raise NetworkBlocked(f"network blocked in File 21 tests: {address!r}")
    return _real_create_connection(address, *args, **kwargs)


def block() -> None:
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.create_connection = _create_connection


def unblock() -> None:
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
    socket.create_connection = _real_create_connection
