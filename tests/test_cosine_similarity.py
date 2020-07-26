import os
import sys
import inspect
import unittest

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

# import app.cosine_similarity


class TestCalc(unittest.TestCase):

    def test_add(self):
        # self.assertEqual(calc.add(10, 5), 15)
        pass

if __name__ == '__main__':
    unittest.main()