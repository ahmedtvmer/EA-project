# Coevolutionary Recommendation Engine

Adaptive recommendation system using coevolutionary algorithms for matrix factorization on the MovieLens 100K dataset.

## Quick Start

```bash
# Install dependencies
uv sync

# Run with dev dependencies (for testing)
uv sync --group dev

# Run the Streamlit app
uv run streamlit run app.py

# Run tests
uv run pytest tests/ -v

# Run EA standalone (quick test)
uv run python ea.py
```

## Features

- **Cooperative Coevolution**: Separate populations for user and item latent factors, evolved cooperatively
- **Competitive Coevolution**: Two parallel full-solution populations competing on fitness
- **Live Training Visualization**: RMSE convergence plotted in real-time
- **Personalized Recommendations**: Top-N movie recommendations for any user
- **Baseline Comparison**: Global-mean RMSE displayed for comparison

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   app.py    │────▶│    ea.py    │────▶│   data.py   │
│  Streamlit  │    │  EA Core    │     │   Data IO   │
│     UI      │     │  Algorithm  │     │  + Matrix   │
└─────────────┘     └─────────────┘     └─────────────┘
```

- **`data.py`**: Loads MovieLens dataset, builds rating matrix, train/test split, RMSE fitness
- **`ea.py`**: Coevolutionary algorithm core (CooperativeEA, CompetitiveEA, operators)
- **`app.py`**: Streamlit UI for interactive training and recommendations

## Algorithm

### Cooperative Strategy
- User population: latent factor vectors for all users
- Item population: latent factor vectors for all items
- Fitness pairing: best-of-population (each user evaluated against best item vector, vice versa)
- Output: `user_factors @ item_factors.T`

### Competitive Strategy
- Two parallel populations, each containing full solutions (user + item factors)
- Competition: best individual from each population competes; winner gets more elite slots
- Output: best solution across both populations

### Operators
- **Selection**: Tournament (size=3)
- **Crossover**: Blend (BLX-α, α=0.5, rate=0.8)
- **Mutation**: Gaussian (scale=0.1, rate=0.1)
- **Elitism**: Top 2 preserved per generation

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Latent Dimensions | 20 | Dimensionality of user/item latent vectors |
| Population Size | 50 | Number of individuals per population |
| Generations | 50 | Number of EA generations |
| Crossover Rate | 0.8 | Probability of crossover |
| Mutation Rate | 0.1 | Probability of mutation per gene |
| Mutation Scale | 0.1 | Standard deviation of Gaussian mutation |
| Tournament Size | 3 | Number of individuals in tournament |
| Elite Size | 2 | Number of elites preserved |

## Dataset

MovieLens 100K:
- 943 users
- 1,682 items (movies)
- 100,000 ratings (1-5 scale)
- Sparsity: ~93.7%

Auto-downloaded from [GroupLens](https://files.grouplens.org/datasets/movielens/ml-100k.zip) on first run.

## Success Criteria

- Cooperative EA achieves RMSE < 1.10 on test set (baseline: ~1.13)
- Competitive EA achieves RMSE within 5% of cooperative
- Training (50 generations) completes in < 30 seconds
- Reproducible results with fixed seed

## Development

This project follows the Spec Kit methodology. See:

- `specs/001-ea-recommender/spec.md` - Feature specification
- `specs/001-ea-recommender/plan.md` - Implementation plan
- `specs/001-ea-recommender/tasks.md` - Task breakdown
- `.specify/memory/constitution.md` - Project constitution

## License

MIT
