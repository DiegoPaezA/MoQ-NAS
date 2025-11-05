# In algorithms/ga/operators.py

import numpy as np

def hux_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs Half-Uniform Crossover (HUX) on two parents.

    HUX identifies differing genes and swaps exactly half of them.

    Args:
        parent1 (np.ndarray): The first parent chromosome.
        parent2 (np.ndarray): The second parent chromosome.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the two offspring.
    """
    differing_indices = np.where(parent1 != parent2)[0]
    num_swaps = len(differing_indices) // 2
    swap_indices = np.random.choice(differing_indices, num_swaps, replace=False)
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    offspring1[swap_indices], offspring2[swap_indices] = parent2[swap_indices], parent1[swap_indices]
    return offspring1, offspring2

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs Uniform Crossover on two parents.

    A binary mask is created, and genes are swapped between parents
    where the mask is True.

    Args:
        parent1 (np.ndarray): The first parent chromosome.
        parent2 (np.ndarray): The second parent chromosome.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the two offspring.
    """
    chromosome_length = len(parent1)
    crossover_mask = np.random.randint(0, 2, size=chromosome_length).astype(bool)
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    offspring1[crossover_mask], offspring2[crossover_mask] = parent2[crossover_mask], parent1[crossover_mask]
    return offspring1, offspring2

def apply_crossover(best_current_pop: np.ndarray, new_pop: np.ndarray, method: str) -> np.ndarray:
    """Applies the selected crossover method to generate offspring.

    It pairs individuals from `best_current_pop` and `new_pop` as parents.

    Args:
        best_current_pop (np.ndarray): Best individuals from the current population.
        new_pop (np.ndarray): Individuals from the newly generated population.
        method (str): The name of the method ('hux' or 'uniform').

    Returns:
        np.ndarray: A population of offspring.
    """
    offspring = []
    for parent1, parent2 in zip(best_current_pop, new_pop):
        if method == 'hux':
            child1, child2 = hux_crossover(parent1, parent2)
        elif method == 'uniform':
            child1, child2 = uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {method}")
        offspring.extend([child1, child2])
    return np.array(offspring[:len(new_pop)])