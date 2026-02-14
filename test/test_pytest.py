from src.titanic_model import load_data, train_model


def test_load_data():
    X, y = load_data()

    assert len(X) > 0
    assert len(X) == len(y)
    assert X.shape[1] == 5  # Pclass, Age, SibSp, Parch, Fare


def test_model_accuracy():
    _, acc = train_model()
    assert acc > 0.70
