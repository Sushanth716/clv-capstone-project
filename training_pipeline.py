import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(path):
    return pd.read_csv(path)

def train_model(df):
    X = df.drop(["Customer ID", "CLV_Segment"], axis=1)
    y = df["CLV_Segment"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    df = load_data("../data/feature_engineered_data.csv")
    model = train_model(df)
    save_model(model, "../model/clv_model.pkl")