import numpy as np
import random
import coevolution_base as cb
import data as dt

ELITISM_COUNT = 2
STRATEGY_COOPERATIVE = "cooperative"
STRATEGY_COMPETITIVE  = "competitive"


def decode_predictions(user_pop, item_pop):
    # dot product between users and items to get the predicted ratings
    # then scale it from [0,1] to [1,5] so it matches the dataset
    U = np.array(user_pop, dtype=np.float32)
    V = np.array(item_pop, dtype=np.float32)
    raw = U @ V.T
    scaled = raw * 4.0 + 1.0
    return np.clip(scaled, 1.0, 5.0)


def _elites(population, scores, n=ELITISM_COUNT):
    # sort by score and return the best n individuals
    # we do this so the good ones dont get lost in the next gen
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [population[i][:] for i in ranked[:n]]


def _cooperative_fitness(predicted, train_matrix):
    # calculate fitness for users and items
    # lower RMSE = better prediction = higher fitness
    n_users, n_items = train_matrix.shape
    eps = 1e-6

    user_fitness = []
    for i in range(n_users):
        mask = train_matrix[i, :] > 0
        if mask.sum() > 0:
            rmse = dt.calculate_fitness(predicted[i, :], train_matrix[i, :], mask=mask)
        else:
            rmse = 5.0  # no ratings for this user so assume worst case
        user_fitness.append(1.0 / (rmse + eps))

    item_fitness = []
    for j in range(n_items):
        mask = train_matrix[:, j] > 0
        if mask.sum() > 0:
            rmse = dt.calculate_fitness(predicted[:, j], train_matrix[:, j], mask=mask)
        else:
            rmse = 5.0
        item_fitness.append(1.0 / (rmse + eps))

    return user_fitness, item_fitness


def _competitive_fitness(predicted, train_matrix, top_k=10):
    # here items compete against each other
    # the item that appears more in top-k recommendations gets higher fitness
    n_users, n_items = train_matrix.shape
    eps = 1e-6

    # users still use rmse like before
    user_fitness = []
    for i in range(n_users):
        mask = train_matrix[i, :] > 0
        if mask.sum() > 0:
            rmse = dt.calculate_fitness(predicted[i, :], train_matrix[i, :], mask=mask)
        else:
            rmse = 5.0
        user_fitness.append(1.0 / (rmse + eps))

    # count how many times each item got recommended
    recommendation_counts = np.zeros(n_items, dtype=np.float32)
    for i in range(n_users):
        unrated = train_matrix[i, :] == 0
        scores_for_user = predicted[i, :].copy()
        scores_for_user[~unrated] = -np.inf  # skip movies the user already rated
        top_k_idx = np.argsort(scores_for_user)[::-1][:top_k]
        recommendation_counts[top_k_idx] += 1.0

    max_count = recommendation_counts.max() if recommendation_counts.max() > 0 else 1.0

    # item_fitness = (recommendation_counts / max_count).tolist()

    # Log-scale popularity to reduce data skewness
    log_pop = np.log1p(recommendation_counts) / np.log1p(max_count)

    # Blend popularity with RMSE-based fitness to provide gradient-like
    item_rmse_fitness = []
    for j in range(n_items):
        mask = train_matrix[:, j] > 0
        if mask.sum() > 0:
            rmse = dt.calculate_fitness(predicted[:, j], train_matrix[:, j], mask=mask)
        else:
            rmse = 5.0
        item_rmse_fitness.append(1.0 / (rmse + eps))

    alpha = 0.5
    item_fitness = (
        alpha * log_pop + (1 - alpha) * np.array(item_rmse_fitness)
    ).tolist()

    item_fitness = [max(f, 0.01) for f in item_fitness]  # dont want it to be zero

    return user_fitness, item_fitness


def coevolution_step(user_pop, item_pop, train_matrix, strategy=STRATEGY_COOPERATIVE):
    # one full generation: evaluate then evolve both populations
    predicted = decode_predictions(user_pop, item_pop)

    if strategy == STRATEGY_COMPETITIVE:
        user_fitness, item_fitness = _competitive_fitness(predicted, train_matrix)
    else:
        user_fitness, item_fitness = _cooperative_fitness(predicted, train_matrix)

    # save the best ones before evolving so we dont lose them
    user_elites = _elites(user_pop, user_fitness)
    item_elites = _elites(item_pop, item_fitness)

    new_user_pop = cb.next_generation(user_pop, user_fitness)
    new_item_pop = cb.next_generation(item_pop, item_fitness)

    # put elites back at the end of the new population
    for k, elite in enumerate(user_elites):
        new_user_pop[-(k + 1)] = elite
    for k, elite in enumerate(item_elites):
        new_item_pop[-(k + 1)] = elite

    return new_user_pop, new_item_pop, user_fitness, item_fitness


def run_coevolution(train_matrix, test_matrix,
                    n_generations=50,
                    strategy=STRATEGY_COOPERATIVE,
                    seed=42,
                    verbose=True):

    random.seed(seed)
    np.random.seed(seed)

    n_users, n_items = train_matrix.shape
    latent_dim = cb.GENOME_LENGTH

    # start with random populations
    user_pop = [cb.create_individual() for _ in range(n_users)]
    item_pop = [cb.create_individual() for _ in range(n_items)]

    history = []
    test_mask  = test_matrix > 0
    train_mask = train_matrix > 0

    if verbose:
        print("=" * 60)
        print(f"  Coevolution  |  strategy={strategy}  |  k={latent_dim}")
        print(f"  {n_users} users  x  {n_items} items  |  {n_generations} generations")
        print("=" * 60)

    for gen in range(1, n_generations + 1):

        user_pop, item_pop, u_fit, i_fit = coevolution_step(
            user_pop, item_pop, train_matrix, strategy=strategy
        )

        predicted = decode_predictions(user_pop, item_pop)

        train_rmse = (dt.calculate_fitness(predicted, train_matrix, mask=train_mask)
                      if train_mask.sum() > 0 else 0.0)
        test_rmse  = (dt.calculate_fitness(predicted, test_matrix,  mask=test_mask)
                      if test_mask.sum()  > 0 else train_rmse)

        avg_u = float(np.mean(u_fit))
        avg_i = float(np.mean(i_fit))

        # save history to plot it later
        history.append({
            "generation":        gen,
            "train_rmse":        train_rmse,
            "test_rmse":         test_rmse,
            "avg_user_fitness":  avg_u,
            "avg_item_fitness":  avg_i,
        })

        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"  Gen {gen:>4} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Test RMSE: {test_rmse:.4f} | "
                  f"Avg User Fit: {avg_u:.4f} | "
                  f"Avg Item Fit: {avg_i:.4f}")

    if verbose:
        print("\n Coevolution complete!")
    return user_pop, item_pop, history


def recommend(user_idx, user_pop, item_pop, item_ids, movie_titles,
              rated_mask, top_n=10):

    predicted = decode_predictions(user_pop, item_pop)
    user_scores = predicted[user_idx, :].copy()

    user_scores[rated_mask] = -np.inf  # dont recomend movies the user already watched

    top_indices = np.argsort(user_scores)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        iid   = int(item_ids[idx])
        title = movie_titles.get(iid, f"Movie ID {iid}")
        pred  = float(np.clip(user_scores[idx], 1.0, 5.0))
        results.append({
            "rank":             rank,
            "item_id":          iid,
            "title":            title,
            "predicted_rating": round(pred, 2),
        })

    return results


if __name__ == "__main__":
    print("Loading data ...")
    matrix, user_ids, item_ids, titles = dt.get_rating_matrix()

    small_matrix = matrix[:50, :100]
    train, test  = dt.train_test_split_matrix(small_matrix)

    print("\n--- Testing COOPERATIVE strategy ---")
    u_pop, i_pop, hist_coop = run_coevolution(
        train, test, n_generations=20, strategy=STRATEGY_COOPERATIVE
    )

    print("\n--- Testing COMPETITIVE strategy ---")
    u_pop2, i_pop2, hist_comp = run_coevolution(
        train, test, n_generations=20, strategy=STRATEGY_COMPETITIVE
    )

    print("\nCooperative final test RMSE :", hist_coop[-1]["test_rmse"])
    print("Competitive  final test RMSE :", hist_comp[-1]["test_rmse"])
