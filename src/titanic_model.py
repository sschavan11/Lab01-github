import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "data/Titanic-Dataset.csv"  # must match your exact filename

def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)

    df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].dropna()

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y

def train_model(random_state: int = 42):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc
