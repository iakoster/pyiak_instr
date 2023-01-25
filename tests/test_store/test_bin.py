import unittest

from pyinstr_iakoster.store import BinaryData

from tests.env_vars import TEST_DATA_DIR


TEST_BIN_DATA = TEST_DATA_DIR / "data_bin"


class TestBinaryData(unittest.TestCase):

    CONTENT = (
        b"\x00\x62\x69\x6e\x00\x00\x00\xa5\x0a\xff\xfa\x40\x2d\xf8\x54\xdd"
    )

    @classmethod
    def setUpClass(cls) -> None:
        TEST_BIN_DATA.mkdir(parents=True, exist_ok=True)
        with open(TEST_BIN_DATA / "ref.bin", "wb") as file:
            file.write(cls.CONTENT)

    def test_init(self) -> None:
        bd = BinaryData(self.CONTENT)
        self.assertEqual(self.CONTENT, bd.content)

    def test_read(self) -> None:
        bd = BinaryData.read(TEST_BIN_DATA / "ref.bin")
        self.assertEqual(self.CONTENT, bd.content)
