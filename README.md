** A-star-Algorithm-for-15-puzzle**
A program that generates 100 random 15-puzzles and solves them using the A* algorithm with 3 different heuristics.

##h1(n) = Number of misplaced tiles.
##h2(n) = Total Manhattan distance (sum of the distances of the tiles from their goal positions)
##h3(n) = Linear Conflict (Manhattan distance with a penalty for tiles that are in the correct row or column but in the wrong order)

The results are printed into a table for performance comparison.
