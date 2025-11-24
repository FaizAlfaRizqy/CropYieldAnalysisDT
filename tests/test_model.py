import unittest
from src.model import DecisionTreeModel
import numpy as np

class TestDecisionTreeModel(unittest.TestCase):

    def setUp(self):
        self.model = DecisionTreeModel()
        self.X_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_train = np.array([1, 2, 3])
        self.X_test = np.array([[2, 3], [4, 5]])

    def test_training(self):
        self.model.train(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.trained_model)

    def test_prediction(self):
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

if __name__ == '__main__':
    unittest.main()