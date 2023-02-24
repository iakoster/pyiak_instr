import unittest

from .env import ROOT_DIR


all_tests = unittest.TestSuite()
all_tests.addTests(
    unittest.TestLoader().discover(
        ROOT_DIR / "tests/test_pyiak_instr",
        top_level_dir=ROOT_DIR,
    )
)
