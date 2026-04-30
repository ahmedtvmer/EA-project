"""
====================================================
 AI420 – Evolutionary Algorithms  |  Spring 2025-2026
 Project: Adaptive Recommendation Engine
          using Coevolutionary Algorithms
====================================================

This file implements the BASE EA LOOP.
Two populations co-evolve together:
  - Population 1: USER profiles  (what users like)
  - Population 2: ITEM profiles  (how items are represented)

Both populations improve each other over generations,
just like predator/prey evolve together in nature.
"""

import random
import numpy as np

# ──────────────────────────────────────────────
# 1.  CONFIGURATION  (easy to tweak)
# ──────────────────────────────────────────────

NUM_USERS        = 10    # number of user "individuals" in the population
NUM_ITEMS        = 10    # number of item "individuals" in the population
GENOME_LENGTH    = 5     # how many preference dimensions each individual has
NUM_GENERATIONS  = 50    # how many evolution cycles to run
MUTATION_RATE    = 0.1   # probability of mutating each gene
CROSSOVER_RATE   = 0.7   # probability of doing crossover vs. cloning
TOURNAMENT_SIZE  = 3     # how many candidates compete in selection


# ──────────────────────────────────────────────
# 2.  REPRESENTATION
#     Each individual is a list of floats [0, 1]
#     representing a feature/preference vector.
# ──────────────────────────────────────────────

def create_individual():
    """Create one random individual (user or item)."""
    return [random.uniform(0, 1) for _ in range(GENOME_LENGTH)]


def create_population(size):
    """Create a full population of `size` individuals."""
    return [create_individual() for _ in range(size)]


# ──────────────────────────────────────────────
# 3.  FITNESS FUNCTION
#     How well does a user–item pair match?
#     Higher score = better recommendation.
# ──────────────────────────────────────────────

def fitness(user, item):
    """
    Dot-product similarity between a user vector and an item vector.
    Think of it as: 'how aligned are their preferences?'
    Returns a value in [0, 1] (normalised).
    """
    score = sum(u * i for u, i in zip(user, item))
    max_possible = GENOME_LENGTH  # all genes = 1.0
    return score / max_possible


def population_fitness(users, items):
    """
    Evaluate every user against every item.
    Returns:
      user_scores  – average fitness of each user across all items
      item_scores  – average fitness of each item across all users
    """
    user_scores = []
    for user in users:
        avg = sum(fitness(user, item) for item in items) / len(items)
        user_scores.append(avg)

    item_scores = []
    for item in items:
        avg = sum(fitness(user, item) for user in users) / len(users)
        item_scores.append(avg)

    return user_scores, item_scores


# ──────────────────────────────────────────────
# 4.  SELECTION
#     Tournament selection: pick the best out of
#     a random small group (a "tournament").
# ──────────────────────────────────────────────

def tournament_select(population, scores):
    """
    Randomly pick TOURNAMENT_SIZE individuals,
    return the one with the highest fitness score.
    """
    contestants = random.sample(range(len(population)), TOURNAMENT_SIZE)
    winner = max(contestants, key=lambda idx: scores[idx])
    return population[winner][:]   # return a copy


# ──────────────────────────────────────────────
# 5.  CROSSOVER
#     Combine two parents to create two children.
#     Single-point crossover: swap gene tails.
# ──────────────────────────────────────────────

def crossover(parent_a, parent_b):
    """
    If crossover happens, swap genes after a random cut point.
    Otherwise, return copies of the parents unchanged.
    """
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENOME_LENGTH - 1)
        child_a = parent_a[:point] + parent_b[point:]
        child_b = parent_b[:point] + parent_a[point:]
        return child_a, child_b
    return parent_a[:], parent_b[:]


# ──────────────────────────────────────────────
# 6.  MUTATION
#     Randomly nudge some genes by a small amount.
# ──────────────────────────────────────────────

def mutate(individual):
    """
    For each gene, with probability MUTATION_RATE,
    add a small random noise and clamp to [0, 1].
    """
    return [
        min(1.0, max(0.0, gene + random.gauss(0, 0.1)))
        if random.random() < MUTATION_RATE
        else gene
        for gene in individual
    ]


# ──────────────────────────────────────────────
# 7.  NEXT GENERATION
#     Build a new population using selection,
#     crossover, and mutation.
# ──────────────────────────────────────────────

def next_generation(population, scores):
    """
    Produce a new population of the same size.
    Uses: tournament selection → crossover → mutation.
    """
    new_pop = []
    while len(new_pop) < len(population):
        parent_a = tournament_select(population, scores)
        parent_b = tournament_select(population, scores)
        child_a, child_b = crossover(parent_a, parent_b)
        new_pop.append(mutate(child_a))
        if len(new_pop) < len(population):
            new_pop.append(mutate(child_b))
    return new_pop


# ──────────────────────────────────────────────
# 8.  MAIN COEVOLUTIONARY LOOP
# ──────────────────────────────────────────────

def run_coevolution():
    """
    The heart of the project.
    Two populations (users & items) evolve in parallel,
    each improving based on how well they interact with the other.
    """

    print("=" * 50)
    print("  Coevolutionary Recommendation Engine")
    print("=" * 50)

    # --- Initialise both populations ---
    users = create_population(NUM_USERS)
    items = create_population(NUM_ITEMS)

    history = []   # track average fitness over time

    for generation in range(1, NUM_GENERATIONS + 1):

        # --- Evaluate fitness (co-dependently) ---
        user_scores, item_scores = population_fitness(users, items)

        avg_user_fit = sum(user_scores) / len(user_scores)
        avg_item_fit = sum(item_scores) / len(item_scores)
        history.append((generation, avg_user_fit, avg_item_fit))

        # --- Print progress every 10 generations ---
        if generation % 10 == 0 or generation == 1:
            print(f"Gen {generation:>3} | "
                  f"Avg User Fitness: {avg_user_fit:.4f} | "
                  f"Avg Item Fitness: {avg_item_fit:.4f}")

        # --- Evolve both populations ---
        users = next_generation(users, user_scores)
        items = next_generation(items, item_scores)

    print("\n✅ Evolution complete!")
    return users, items, history


# ──────────────────────────────────────────────
# 9.  RECOMMENDATION  (simple demo)
# ──────────────────────────────────────────────

def recommend_for_user(user, items, top_n=3):
    """
    Given a user vector, rank all items by fitness score
    and return the top N recommended item indices.
    """
    scores = [(idx, fitness(user, item)) for idx, item in enumerate(items)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ──────────────────────────────────────────────
# 10. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # Run the coevolutionary loop
    final_users, final_items, history = run_coevolution()

    # Demo: show recommendations for the first user
    print("\n📋 Sample recommendations for User 0:")
    recs = recommend_for_user(final_users[0], final_items, top_n=3)
    for rank, (item_idx, score) in enumerate(recs, 1):
        print(f"  Rank {rank}: Item {item_idx}  (score: {score:.4f})")
