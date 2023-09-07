"""Private module of ``pyiak_instr.utilities``"""
from ..codecs.types import Codec


__all__ = ["BasicByteStuffingCodec"]


class BasicByteStuffingCodec(
    Codec[tuple[bytes, dict[int, bytes]], bytes, bytes]
):
    """
    Codec for byte stuffing.

    Parameters
    ----------
    stuff_byte : bytes
        byte to be stuffed.
    stuff_length : int
        number of bytes to be stuffed.

    Raises
    ------
    ValueError
        * if 'stuff_byte' is not a single byte;
        * if 'stuff_length' is less than 1.
    """

    def __init__(self, stuff_byte: bytes = b"\x00", stuff_length: int = 1):
        if len(stuff_byte) != 1:
            raise ValueError("'stuff_byte' must be a single byte")
        if stuff_length < 1:
            raise ValueError("'stuff_length' must be at least 1")

        self._s_byte = stuff_byte
        self._len = stuff_length

    def decode(self, data: bytes) -> tuple[bytes, dict[int, bytes]]:
        """
        Decode the given data by performing the inverse of byte stuffing.

        Parameters
        ----------
        data : bytes
            the data to be decoded.

        Returns
        -------
        tuple[bytes, dict[int, bytes]]
            * the decoded bytes;
            * dictionary with indexes and additional stuff bytes.

        Raises
        ------
        ValueError
            if the additional bytes after the stuff byte are not of the
            required length.
        """
        decoded, stuff_bytes, ref = b"", {}, b"\x00" * self._len
        i, shift = 0, 0
        while i < len(data):
            byte = data[i : i + 1]
            if byte == self._s_byte:
                add_bytes = data[i + 1 : i + 1 + self._len]
                if len(add_bytes) != self._len:
                    raise ValueError(
                        "stuffing additional bytes do not have required "
                        "length"
                    )

                if add_bytes != ref:
                    stuff_bytes[i - shift] = add_bytes
                    shift += 1
                else:
                    decoded += byte

                shift += self._len
                i += self._len

            else:
                decoded += byte
            i += 1

        return decoded, stuff_bytes

    def encode(self, data: bytes) -> bytes:
        """
        Encode the given data by performing byte stuffing.

        Parameters
        ----------
        data : bytes
            the input data to be encoded.

        Returns
        -------
        bytes
            the encoded data.
        """
        byte_stuff = b"\x00" * self._len
        result = b""
        for i in range(len(data)):
            byte = data[i : i + 1]
            result += byte
            if byte == self._s_byte:
                result += byte_stuff
        return result
