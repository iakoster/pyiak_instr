

class Register(object):

    def __init__(
            self, start_address: int, length: int,
            word_length: int = 1, word_length_variable: bool = False
    ):
        self._st_addr = start_address
        self._len = length

        self._w_len_var = word_length_variable
        self._w_len = 1 if self._w_len_var else word_length

    @property
    def start_address(self):
        return self._st_addr

    @property
    def length(self):
        return self._len

    @property
    def word_length(self):
        return self._w_len

    @property
    def word_length_variable(self):
        return self._w_len_var
