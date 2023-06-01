import unittest

from src.pyiak_instr.communication.message import (
    MessagePattern,
    MessageStructPattern,
    MessageFieldStructPattern,
)
from src.pyiak_instr.communication.format_map import PatternsMap

from ....utils import validate_object
from ...env import (
    get_local_test_data_dir,
    remove_test_data_dir,
)


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestPatternsMap(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            pattern_names=["s0", "s1"],
        )

    def test__get_pattern_names(self) -> None:
        path = TEST_DATA_DIR / "test__get_pattern_names.ini"
        obj = self._instance()
        obj.write(path)
        self.assertListEqual(["s0", "s1"], obj._get_pattern_names(path))

    @staticmethod
    def _instance() -> PatternsMap:
        return PatternsMap(
            MessagePattern.basic().configure(
                s0=MessageStructPattern.basic().configure(
                    f0=MessageFieldStructPattern.static(),
                    f1=MessageFieldStructPattern.data(),
                )
            ),
            MessagePattern.basic().configure(
                s1=MessageStructPattern.basic().configure(
                    f0=MessageFieldStructPattern.id_(),
                    f1=MessageFieldStructPattern.dynamic_length(),
                    f2=MessageFieldStructPattern.crc(),
                )
            ),
        )
