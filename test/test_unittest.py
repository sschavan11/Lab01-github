import unittest
from src.titanic_model import load_data, train_model


class TestTitanicModel(unittest.TestCase):

    def test_load_data(self):
        X, y = load_data()
        self.assertTrue(len(X) > 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 5)

    def test_model_accuracy(self):
        _, acc = train_model()
        self.assertTrue(acc > 0.70)


if __name__ == "__main__":
    unittest.main()
