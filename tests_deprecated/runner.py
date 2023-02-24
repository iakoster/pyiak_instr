import unittest as _unittest
from pathlib import Path as _Path


_tests_loader = _unittest.TestLoader()
all_tests_suite = _unittest.TestSuite()
all_tests_suite.addTests(_tests_loader.discover(str(_Path('.\\tests')), top_level_dir='../tests'))
