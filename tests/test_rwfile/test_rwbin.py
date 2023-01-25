import shutil
import unittest

from pyinstr_iakoster.rwfile import RWBin

from tests.env_vars import TEST_DATA_DIR

TEST_RWBIN_DATA = TEST_DATA_DIR / "data_rwbin"


class TestRWBin(unittest.TestCase):

    CONTENT = b"\x00\x62\x69\x6e\x00\x00\x00\xa5\x0a\xff\xfa\x40\x2d\xf8\x54\xdd"
    # bin, 165(>I), 2815(>H), 250(B), 2.71828174591064453125(raw 2.718281828, >f), 221(B)
    # bin, [0, 0, 0, 165, 10, 255, 250, 64, 45, 248, 84, 221](B)

    REF_FILE = TEST_RWBIN_DATA / "ref.bin"

    @classmethod
    def setUpClass(cls) -> None:
        TEST_RWBIN_DATA.mkdir(parents=True, exist_ok=True)
        with open(cls.REF_FILE, "wb") as file:
            file.write(cls.CONTENT)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_DIR)

    def test_write(self) -> None:
        path = TEST_RWBIN_DATA / "res.bin"
        with RWBin(path) as rwb:
            rwb.rewrite(self.CONTENT)

        with open(self.REF_FILE, "rb") as ref, open(path, "rb") as res:
            self.assertEqual(ref.read(), res.read())

    def test_write_read(self) -> None:
        path = TEST_RWBIN_DATA / "res.bin"
        with RWBin(path) as rwb:
            rwb.rewrite(self.CONTENT)
            res = rwb.read_all()

        self.assertEqual(self.CONTENT, res)
