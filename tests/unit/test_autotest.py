'''
    test continuous integration test enviroment
'''

import unittest
import pgdrome

class Test(unittest.TestCase):
    def setUp(self):
        self.a = 2


    def test_a(self):
        self.assertAlmostEqual(2, self.a, places=7)


if __name__ == "__main__":
    unittest.main()
