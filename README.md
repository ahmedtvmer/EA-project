# Coevolutionary Recommendation Engine

Adaptive recommendation system using coevolutionary algorithms for matrix factorization on the MovieLens 100K dataset.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run the coevolution engine standalone (tests both strategies)
python coevolution_engine.py

# Run the base EA operators standalone
python coevolution_base.py

# Run the data loader standalone
python data.py
```

## Features

- **Cooperative Coevolution**: Users and items evolve together, each evaluated against the best of the other population using RMSE-based fitness
- **Competitive Coevolution**: Items compete for recommendation slots — fitness based on how often they appear in top-k recommendations
- **Side-by-Side Comparison**: Both strategies run automatically; overlaid RMSE curves and comparison table displayed
- **Live Training Visualization**: RMSE convergence plotted in real-time
- **Personalized Recommendations**: Top-N movie recommendations for any user
- **5-Star History**: Shows the target user's actual 5-star rated movies as a trust anchor
- **Baseline Comparison**: Global-mean RMSE displayed for comparison

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│   app.py    │────▶│ coevolution_engine  │────▶│ coevolution_base  │
│  Streamlit  │     │  (both strategies)  │     │  (EA operators)   │
│     UI      │     │                     │     │                   │
└──────┬──────┘     └──────────┬──────────┘     └───────────────────┘
       │                       │
       └───────────────────────┼─────────────────────────────────────┐
                               ▼                                     ▼
                       ┌───────────────┐                     ┌───────────────┐
                       │   data.py     │                     │   data.py     │
                       │  (data IO)    │                     │  (fitness)    │
                       └───────────────┘                     └───────────────┘
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `data.py` | Loads MovieLens 100K dataset, builds rating matrix, train/test split, RMSE fitness calculation |
| `coevolution_base.py` | Core EA operators: individual creation, tournament selection, single-point crossover, Gaussian mutation, next generation |
| `coevolution_engine.py` | Coevolutionary strategies: cooperative fitness, competitive fitness, elitism, full training loop, recommendation generation |
| `app.py` | Streamlit UI: parameter controls, live RMSE chart, strategy comparison table, recommendations display |

## Algorithm

### Cooperative Strategy
- **Fitness**: Per-user and per-item RMSE on training set, inverted to maximization (`1/(rmse + eps)`)
- **Pairing**: Each user/item evaluated against the current predicted matrix
- **Evolution**: Tournament selection → crossover → mutation → elitism (top 2 preserved)

### Competitive Strategy
- **User fitness**: Same RMSE-based as cooperative
- **Item fitness**: Based on how often each item appears in top-k recommendations across all users — items compete for visibility
- **Evolution**: Same operators; competition drives items to be more discriminative

### EA Operators (from `coevolution_base.py`)

| Operator | Type | Parameters |
|----------|------|------------|
| Selection | Tournament | Size = 3 |
| Crossover | Single-point | Rate = 0.7 |
| Mutation | Gaussian noise | Rate = 0.1, σ = 0.1 |
| Elitism | Top-N preservation | Count = 2 |
| Genome | Float vector [0, 1] | Length = 5 |

## Configuration

| Parameter | Default | Location |
|-----------|---------|----------|
| Genome Length (latent dim) | 5 | `coevolution_base.py` |
| Mutation Rate | 0.1 | `coevolution_base.py` |
| Crossover Rate | 0.7 | `coevolution_base.py` |
| Tournament Size | 3 | `coevolution_base.py` |
| Elitism Count | 2 | `coevolution_engine.py` |
| Generations | 50 (UI slider) | `app.py` |

## Dataset

**MovieLens 100K** (publicly available from [GroupLens](https://grouplens.org/datasets/movielens/100k/)):
- 943 users
- 1,682 items (movies)
- 100,000 ratings (1-5 scale)
- Sparsity: ~93.7%

Auto-downloaded from GroupLens on first run. Stored in `ml-100k/` directory.

## Project Structure

```
.
├── app.py                  # Streamlit UI
├── coevolution_base.py     # EA operators (selection, crossover, mutation)
├── coevolution_engine.py   # Coevolution strategies (cooperative + competitive)
├── data.py                 # MovieLens data loading and RMSE fitness
├── requirements.txt        # Python dependencies
├── ml-100k/                # Dataset (auto-downloaded)
├── project_instructions/   # Course assignment documents
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Dependencies

- Python 3.12+
- numpy>=2.4.4
- pandas>=3.0.2
- streamlit>=1.56.0

Install via `pip install -r requirements.txt`.

## Course

AI420 – Evolutionary Algorithms | Spring 2025-2026  
Capital University, Faculty of Computing & Artificial Intelligence  
Project [10]: Adaptive Recommendation Engine using Coevolutionary Algorithms
