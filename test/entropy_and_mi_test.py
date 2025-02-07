import unittest
import numpy as np
from entropy_and_mi import fill_redundancy, fill_importance
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


class TestEntropyAndMi(unittest.TestCase):

    def test_data(self):
        data, _ = load_breast_cancer(return_X_y=True)
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        self.assertEqual(np.round(np.max(normalized_data), 2), 1)
        self.assertEqual(np.round(np.min(normalized_data), 2), 0)

    def test_fill_redundancy(self):
        features = len(data.columns)
        r_matrix_shape = fill_redundancy(data, bins).shape
        self.assertEqual(r_matrix_shape, (features, features))

    def test_fill_importance(self):
        features = len(data.columns)
        i_matrix_shape = fill_importance(data, target, bins).shape
        self.assertEqual(i_matrix_shape, (features, ))


if __name__ == '__main__':
    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    bins = 10
    unittest.main()