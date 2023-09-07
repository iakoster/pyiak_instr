from ..codecs.types import Codec


class ByteStuffingCodec(Codec[tuple[bytes, dict[int, bytes]], bytes, bytes]):

    def __init__(self):
        self._s_byte = b"\x00"
        self._len = 1

    def decode(self, data: bytes) -> tuple[bytes, dict[int, bytes]]:
        ...

    def encode(self, data: bytes) -> bytes:
        byte_stuff = b"\x00" * self._len
        result = b""
        for i in range(len(data)):
            byte = data[i : i + 1]
            result += byte
            if byte == self._s_byte:
                result += byte_stuff
        return result
