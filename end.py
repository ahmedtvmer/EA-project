import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR      = "ml-100k"
RATINGS_FILE  = os.path.join(DATA_DIR, "u.data")


def download_movielens():
    if os.path.exists(DATA_DIR):
        return
    print("Downloading MovieLens 100k...")
    urllib.request.urlretrieve(MOVIELENS_URL, "ml-100k.zip")
    with zipfile.ZipFile("ml-100k.zip", "r") as zf:
        zf.extractall(".")
    os.remove("ml-100k.zip")
    print("Done.")


def load_data():
    if not os.path.exists(RATINGS_FILE):
        download_movielens()
    df = pd.read_csv(RATINGS_FILE, sep="\t",
                     names=["user_id", "item_id", "rating", "timestamp"])
    return df


def get_rating_matrix(df=None):
    if df is None:
        df = load_data()

    user_ids = np.sort(df["user_id"].unique())
    item_ids = np.sort(df["item_id"].unique())

    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {v: j for j, v in enumerate(item_ids)}

    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    for row in df.itertuples(index=False):
        matrix[user_index[row.user_id], item_index[row.item_id]] = row.rating

    return matrix, user_ids, item_ids


def calculate_fitness(predicted, actual, mask=None):
    predicted, actual = np.array(predicted), np.array(actual)
    if mask is not None:
        predicted, actual = predicted[mask], actual[mask]
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def train_test_split_matrix(matrix, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    rows, cols = np.where(matrix > 0)
    test_idx = rng.choice(len(rows), size=int(len(rows) * test_ratio), replace=False)

    train = matrix.copy()
    test  = np.zeros_like(matrix)
    for i in test_idx:
        test[rows[i], cols[i]]  = matrix[rows[i], cols[i]]
        train[rows[i], cols[i]] = 0.0
    return train, test


def get_rated_mask(matrix):
    return matrix > 0


if __name__ == "__main__":
    matrix, users, items = get_rating_matrix()
    train, test = train_test_split_matrix(matrix)
    mask = get_rated_mask(test)

    mean_rating = matrix[matrix > 0].mean()
    naive_pred  = np.full_like(test, mean_rating)
    rmse = calculate_fitness(naive_pred, test, mask=mask)

    print(f"Matrix shape : {matrix.shape}")
    print(f"Baseline RMSE: {rmse:.4f}")
