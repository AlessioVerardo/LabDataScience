import unittest
from inspect import isgenerator
from main import fibo


class TestFibo(unittest.TestCase):

    def test_generator(self):
        self.assertTrue(isgenerator(fibo(3)))

    def test_values(self):
        self.assertEqual(
            [1, 1, 2, 3, 5],
            [x for x in fibo(5)]
        )

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            next(fibo('a'))

if __name__ == '__main__':
    unittest.main()
