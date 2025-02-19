import unittest
from qubo_dimod import create_qubo_matrix
from data_prep import new_data, target


class TestQuboMatrix(unittest.TestCase):

    def test_create_qubo_matrix(self):
        data = new_data
        features = len(data.columns)
        bins = 10
        l = 0.5
        _, qubo_matrix = create_qubo_matrix(data, target, bins, l)
        self.assertEqual(qubo_matrix.shape, (features, features))

    def test_symmetric_matrix(self):
        data = new_data
        bins = 10
        l = 0.5
        _, qubo_matrix = create_qubo_matrix(data, target, bins, l)
        self.assertTrue((qubo_matrix == qubo_matrix.T).all())


if __name__ == '__main__':
    unittest.main()