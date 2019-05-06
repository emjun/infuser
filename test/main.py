import os
import glob
import unittest

from infuser import analysis_main


class TestAnalysisMain(unittest.TestCase):
    """Tests of __main__.analysis_main"""

    def setUp(self) -> None:
        self.test_programs_root = \
            os.getenv("INFUSER_TESTPROGRAMS_ROOT", "../../testprograms")

    def test_parses_test_programs_without_crashing(self):
        print(os.getcwd())
        glob_pattern = os.path.join(self.test_programs_root, "*.py")
        print(glob_pattern)
        for path in glob.glob(glob_pattern, recursive=True):
            with open(path, "r") as fo:
                analysis_main(fo)
