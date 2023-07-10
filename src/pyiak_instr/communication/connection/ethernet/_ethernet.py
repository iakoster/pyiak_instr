"""Private module of ``pyiak_instr.communication.connection.ethernet``."""
import re
from itertools import chain
from collections import namedtuple

import psutil  # todo: optional import
import numpy as np


__all__ = ["IPV4_PATTERN", "IPv4Address", "IPv4"]


IPv4Address = namedtuple("IPv4Address", ["ip", "port"])
IPV4_PATTERN = re.compile(
    r"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)(\.(?!$)|$)){4}$"
)
# not better version (stackoverflow.com/questions/5284147)


class IPv4:
    """
    Class to work with IPv4 connections on PC.
    """

    def addresses(self, ip: str | None = None) -> set[IPv4Address]:
        """
        Returns opened IPv4 connections in this PC.

        If `ip` is not None returns opened connections for specified `id`.
        Returns all opened connections instead.

        Parameters
        ----------
        ip: str | None, default=None
            IPv4 address.

        Returns
        -------
        set of IPV4_ADDRESS
            set of opened connections.
        """
        addrs: set[IPv4Address] = set()
        for addr in chain.from_iterable(
            (con.laddr, con.raddr)
            for con in psutil.net_connections(kind="inet4")
        ):
            if len(addr) == 0:
                continue
            addr_ip, addr_port = addr  # type: ignore[misc]
            if ip is None or ip == addr_ip:
                addrs.add(IPv4Address(ip=addr_ip, port=addr_port))

        return addrs

    def ips(self) -> set[str]:
        """
        Returns all available ip on this PC, except for standard addresses.

        Returns
        -------
        set of str
            available addresses.
        """
        return set(addr.ip for addr in self.addresses())

    def ports(self, ip: str | None = None) -> set[int]:
        """
        Returns set of opened ports of IPv4 sockets on this PC.

        If `ip` is not None returns opened ports for specified `ip`, else all
        opened ports on this PC.

        Parameters
        ----------
        ip: str | None, default=None
            IPv4 address.

        Returns
        -------
        set of int
            opened ports.
        """
        return set(con.port for con in self.addresses(ip=ip))

    def random_available_port(
        self,
        ip: str | None = None,
        max_iter: int = 100,
    ) -> int:
        """
        Returns random available port in range [1024,65535].

        If `ip` is None returned available port for all connections.

        Parameters
        ----------
        ip: str | None, default=None
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

        busy = self.ports(ip=ip)
        for _ in range(max_iter):
            port = np.random.randint(1024, 65535)
            if port not in busy:
                return port

        raise ValueError(
            f"no available port was found in {max_iter} attempts"
        )
