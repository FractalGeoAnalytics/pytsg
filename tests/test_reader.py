import unittest

from src.pytsg import parse_tsg
from src.pytsg.asdreader import asdreader


class TestFileReaders(unittest.TestCase):
    def test_read_package(self):
        folder = r"example_data/SWMB007d"
        tmp_data = parse_tsg.read_package(folder, read_cras_file=True)
        self.assertTrue(hasattr(tmp_data, "nir"))
        self.assertTrue(hasattr(tmp_data, "tir"))
        self.assertTrue(hasattr(tmp_data, "cras"))

    def test_asdreader(self):
        asd = asdreader.reader("src/pytsg/asdreader/data/raw.asd")
        self.assertTrue(hasattr(asd, "md"))


if __name__ == "__main__":
    unittest.main()
