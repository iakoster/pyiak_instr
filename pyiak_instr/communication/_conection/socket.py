import re
import socket
from typing import NamedTuple

import psutil  # todo: optional import
import numpy as np


__all__ = [
    "IPV4_ADDRESS_TYPE",
    "IPV4_PATTERN",
    "get_opened_connections",
    "get_busy_ports",
    "get_available_ips",
    "get_random_available_port",
]


IPV4_ADDRESS_TYPE = NamedTuple("addr", [("ip", str), ("port", int)])
IPV4_PATTERN = re.compile("^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)(\.(?!$)|$)){4}$")
# not better version (stackoverflow.com/questions/5284147)


def get_opened_connections(ip: str = None) -> set[IPV4_ADDRESS_TYPE]:
    """
    Returns opened connections in this PC.

    If `ip` is not None returns opened connections for specified `id`.
    Returns all opened connections instead.

    Parameters
    ----------
    ip: str, default=None
        IPv4 address.

    Returns
    -------
    set of IPV4_ADDRESS_TYPE
        set of opened connections.
    """

    def add_if_ip_correct(addr: IPV4_ADDRESS_TYPE) -> None:
        if len(addr) and IPV4_PATTERN.match(addr.ip) is not None \
                and (ip == addr.ip or ip is None):
            addrs.add(addr)

    addrs = set()
    for con in psutil.net_connections(kind="all"):
        add_if_ip_correct(con.laddr)
        add_if_ip_correct(con.raddr)

    return addrs


def get_busy_ports(ip: str = None) -> set[int]:
    """
    Returns set of opened ports on this PC.

    If `ip` is not None returns opened ports for specified `ip`, else all
    opened ports on this PC.

    Parameters
    ----------
    ip: str, default=None
        IPv4 address.

    Returns
    -------
    set of int
        opened ports.
    """
    return set(con.port for con in get_opened_connections(ip=ip))


def get_available_ips() -> set[str]:
    """
    Returns all available ip on this PC, except for standard addresses.

    Returns
    -------
    set of str
        available addresses.
    """
    return set(
        i[4][0] for i in socket.getaddrinfo(
            socket.gethostname(), None
        ) if IPV4_PATTERN.match(i[4][0]) is not None
    )


def get_random_available_port(ip: str = None, max_iter: int = 100) -> int:
    """
    Returns random available port in range [1024,65535].

    If `ip` is None returned available port for all connections.

    Parameters
    ----------
    ip: str, default=None
        ip address where it can try to find available port.
    max_iter: int, default = 100
        max attempts for searching port.

    Returns
    -------
    int
        available port.

    Raises
    ------
    ValueError
        if no available port was found in 100 attempts.
    """
    assert max_iter > 0

    busy = get_busy_ports(ip=ip)
    for _ in range(max_iter):
        port = int(np.random.uniform(1024, 65535))
        if port not in busy:
            return port

    raise ValueError(
        "no available port was found in %d attempts" % max_iter
    )
