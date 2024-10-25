import random
import numpy as np
from heapq import heappop, heappush
from copy import deepcopy
import pandas as pd

# Constants for the 15-puzzle
PUZZLE_SIZE = 4
GOAL_STATE = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 0]]  # 0 represents the empty space

# Directions for tile movements: up, down, left, right
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Helper functions
def is_goal(state):
    """Check if the current state is the goal state."""
    return state == GOAL_STATE

def get_tile_position(state, tile):
    """Get the position of a specific tile in the puzzle."""
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if state[i][j] == tile:
                return (i, j)
    return None

def misplaced_tiles(state):
    """Heuristic h1: Count of misplaced tiles."""
    return sum(1 for i in range(PUZZLE_SIZE) for j in range(PUZZLE_SIZE) if state[i][j] != 0 and state[i][j] != GOAL_STATE[i][j])

def manhattan_distance(state):
    """Heuristic h2: Total Manhattan distance."""
    distance = 0
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if state[i][j] != 0:
                correct_pos = get_tile_position(GOAL_STATE, state[i][j])
                distance += abs(correct_pos[0] - i) + abs(correct_pos[1] - j)
    return distance

def generate_random_puzzle(moves=30):
    """Generate a random reachable puzzle by performing a series of random moves from the goal state."""
    current_state = deepcopy(GOAL_STATE)
    empty_pos = (3, 3)  # Start from the solved position
    for _ in range(moves):
        valid_moves = []
        for direction in DIRECTIONS:
            new_pos = (empty_pos[0] + direction[0], empty_pos[1] + direction[1])
            if 0 <= new_pos[0] < PUZZLE_SIZE and 0 <= new_pos[1] < PUZZLE_SIZE:
                valid_moves.append(direction)
        if valid_moves:
            move = random.choice(valid_moves)
            new_pos = (empty_pos[0] + move[0], empty_pos[1] + move[1])
            # Swap the empty space with the selected tile
            current_state[empty_pos[0]][empty_pos[1]], current_state[new_pos[0]][new_pos[1]] = current_state[new_pos[0]][new_pos[1]], current_state[empty_pos[0]][empty_pos[1]]
            empty_pos = new_pos
    return current_state

def get_neighbors(state):
    """Generate all possible moves (neighbors) from the current state."""
    neighbors = []
    empty_pos = get_tile_position(state, 0)
    
    for direction in DIRECTIONS:
        new_pos = (empty_pos[0] + direction[0], empty_pos[1] + direction[1])
        if 0 <= new_pos[0] < PUZZLE_SIZE and 0 <= new_pos[1] < PUZZLE_SIZE:
            new_state = deepcopy(state)
            # Swap the empty space with the selected tile
            new_state[empty_pos[0]][empty_pos[1]], new_state[new_pos[0]][new_pos[1]] = new_state[new_pos[0]][new_pos[1]], new_state[empty_pos[0]][empty_pos[1]]
            neighbors.append(new_state)
    
    return neighbors

def linear_conflict(state):
    """Heuristic h3: Manhattan distance + Linear Conflict."""
    # Start with the Manhattan distance
    total_distance = manhattan_distance(state)
    
    # Check for linear conflicts in rows
    for row in range(PUZZLE_SIZE):
        max_val_in_row = [-1] * PUZZLE_SIZE  # Initialize to track max in row
        for col in range(PUZZLE_SIZE):
            tile = state[row][col]
            if tile != 0:
                correct_row, correct_col = divmod(tile - 1, PUZZLE_SIZE)
                if correct_row == row:  # Tile is in its correct row
                    for k in range(col + 1, PUZZLE_SIZE):
                        next_tile = state[row][k]
                        if next_tile != 0:
                            correct_next_row, correct_next_col = divmod(next_tile - 1, PUZZLE_SIZE)
                            if correct_next_row == row and correct_col > correct_next_col:
                                total_distance += 2  # Linear conflict detected

    # Check for linear conflicts in columns
    for col in range(PUZZLE_SIZE):
        max_val_in_col = [-1] * PUZZLE_SIZE  # Initialize to track max in column
        for row in range(PUZZLE_SIZE):
            tile = state[row][col]
            if tile != 0:
                correct_row, correct_col = divmod(tile - 1, PUZZLE_SIZE)
                if correct_col == col:  # Tile is in its correct column
                    for k in range(row + 1, PUZZLE_SIZE):
                        next_tile = state[k][col]
                        if next_tile != 0:
                            correct_next_row, correct_next_col = divmod(next_tile - 1, PUZZLE_SIZE)
                            if correct_next_col == col and correct_row > correct_next_row:
                                total_distance += 2  # Linear conflict detected

    return total_distance

def a_star_solver(start_state, heuristic_func):
    """A* search algorithm to solve the 15-puzzle."""
    open_list = []
    heappush(open_list, (0 + heuristic_func(start_state), 0, start_state, []))  # (f, g, state, path)
    closed_set = set()
    nodes_expanded = 0
    
    while open_list:
        _, g, current_state, path = heappop(open_list)
        
        if is_goal(current_state):
            return len(path), nodes_expanded  # Return number of steps and nodes expanded
        
        state_tuple = tuple(tuple(row) for row in current_state)
        if state_tuple in closed_set:
            continue
        
        closed_set.add(state_tuple)
        nodes_expanded += 1
        
        for neighbor in get_neighbors(current_state):
            if tuple(tuple(row) for row in neighbor) not in closed_set:
                new_g = g + 1
                new_f = new_g + heuristic_func(neighbor)
                heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))
    
    return None, nodes_expanded  # If no solution is found

# Generate random puzzles and solve them using h1 and h2
random_puzzles = [generate_random_puzzle(moves=30) for _ in range(100)]
results = []

for puzzle in random_puzzles:
    steps_h1, nodes_h1 = a_star_solver(puzzle, misplaced_tiles)
    steps_h2, nodes_h2 = a_star_solver(puzzle, manhattan_distance)
    results.append({"Initial State": puzzle, "Steps h1": steps_h1, "Nodes Expanded h1": nodes_h1, "Steps h2": steps_h2, "Nodes Expanded h2": nodes_h2})

# Solve the same random puzzles using h3
for idx, puzzle in enumerate(random_puzzles):
    steps_h3, nodes_h3 = a_star_solver(puzzle, linear_conflict)
    results[idx].update({"Steps h3": steps_h3, "Nodes Expanded h3": nodes_h3})

# Update the DataFrame with h3 results
updated_results_df = pd.DataFrame(results)

print("A* Algorithm Results for 15-Puzzle")
print(updated_results_df)
