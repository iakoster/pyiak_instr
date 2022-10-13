from ._msg import Message


class Connection(object):  # nodesc

    def __init__(
            self,
            hapi,
    ):
        self._hapi = hapi

    def send(self, message: Message) -> Message:
        ...

    def _send(self, message: bytes) -> bytes:
        raise NotImplementedError()

    @property
    def hapi(self):
        return self._hapi
