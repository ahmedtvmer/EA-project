import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR      = "ml-100k"
RATINGS_FILE  = os.path.join(DATA_DIR, "u.data")
MOVIES_FILE   = os.path.join(DATA_DIR, "u.item")

def _download_if_needed():
    if os.path.exists(DATA_DIR):
        return
    print("Downloading MovieLens 100k ...")
    urllib.request.urlretrieve(MOVIELENS_URL, "ml-100k.zip")
    with zipfile.ZipFile("ml-100k.zip", "r") as zf:
        zf.extractall(".")
    os.remove("ml-100k.zip")
    print("Download complete.")

def load_movie_titles():
    _download_if_needed()
    movies_df = pd.read_csv(
        MOVIES_FILE,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "movie_title"] + [f"_c{i}" for i in range(22)],
    )[["item_id", "movie_title"]]
    return dict(zip(movies_df["item_id"], movies_df["movie_title"]))

def get_rating_matrix():
    _download_if_needed()

    df = pd.read_csv(
        RATINGS_FILE,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    movie_titles = load_movie_titles()

    user_ids = np.sort(df["user_id"].unique())
    item_ids = np.sort(df["item_id"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {v: j for j, v in enumerate(item_ids)}

    # Fill matrix
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    for row in df.itertuples(index=False):
        matrix[user_idx[row.user_id], item_idx[row.item_id]] = row.rating

    return matrix, user_ids, item_ids, movie_titles

def train_test_split_matrix(matrix, test_ratio=0.2, seed=42):
    rng  = np.random.default_rng(seed)
    rows, cols = np.where(matrix > 0)
    test_idx = rng.choice(len(rows), size=int(len(rows) * test_ratio), replace=False)

    train = matrix.copy()
    test  = np.zeros_like(matrix)
    for i in test_idx:
        test[rows[i], cols[i]]  = matrix[rows[i], cols[i]]
        train[rows[i], cols[i]] = 0.0
    return train, test

def calculate_fitness(predicted, actual, mask=None):
    predicted = np.array(predicted, dtype=np.float32)
    actual    = np.array(actual,    dtype=np.float32)

    if mask is not None:
        predicted = predicted[mask]
        actual    = actual[mask]

    return float(np.sqrt(np.mean((predicted - actual) ** 2)))

if __name__ == "__main__":
    matrix, users, items, titles = get_rating_matrix()
    train, test = train_test_split_matrix(matrix)
    mask = test > 0

    mean_rating = matrix[matrix > 0].mean()
    naive_pred  = np.full_like(test, mean_rating)
    rmse = calculate_fitness(naive_pred, test, mask=mask)

    print(f"Matrix shape       : {matrix.shape}  ({len(users)} users × {len(items)} movies)")
    print(f"Total ratings      : {int((matrix > 0).sum())}")
    print(f"Global mean rating : {mean_rating:.4f}")
    print(f"Baseline RMSE      : {rmse:.4f}")

    ratings_count = (matrix > 0).sum(axis=0)
    top10 = np.argsort(ratings_count)[::-1][:10]

    print("\nTop 10 most-rated movies:")
    print(f"{'Rank':<5} {'item_id':<10} {'# ratings':<12} {'Movie Title'}")
    print("-" * 55)
    for rank, idx in enumerate(top10, 1):
        iid   = items[idx]
        title = titles.get(iid, "Unknown")
        print(f"{rank:<5} {iid:<10} {int(ratings_count[idx]):<12} {title}")

