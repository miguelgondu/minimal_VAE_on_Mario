"""
This script implements some grammar checks for Zelda.
"""
from typing import Tuple
import random

import numpy as np

from queue import Queue

from zelda_utils.plotting import encoding


def grammar_check(level: np.ndarray) -> bool:
    onehot = np.load("./data/processed/zelda/onehot.npz")["levels"]
    levels = onehot.argmax(axis=-1)
    _, x, y = np.where(levels == 3)
    possible_door_positions = set([(xi, yi) for xi, yi in zip(x, y)])

    flag_ = has_outer_walls(level)
    flag_ = flag_ and has_doors_or_stairs(level, possible_door_positions)

    doors_in_level = list(get_doors_in_level(level))
    if len(doors_in_level) >= 2:
        for i, door_1 in enumerate(doors_in_level):
            for door_2 in doors_in_level[(i + 1) :]:
                flag_ = flag_ and are_doors_connected(level, door_1, door_2)

    return flag_


def has_outer_walls(level: np.ndarray) -> bool:
    flag_ = (level[:, 0] == 13).all()
    flag_ = flag_ and (level[:, -1] == 13).all()
    flag_ = flag_ and (level[0, :] == 13).all()
    flag_ = flag_ and (level[-1, :] == 13).all()
    return flag_


def get_doors_in_level(level) -> set:
    x, y = np.where(level == 3)
    if len(x) == 0:
        # There are no doors :(
        return set([])

    # Making sure doors are complete
    left_doors = [(4, 1), (5, 1), (6, 1)]
    right_doors = [(4, 14), (5, 14), (6, 14)]
    upper_doors = [(1, 7), (1, 8)]
    lower_doors = [(9, 7), (9, 8)]
    doors_present_in_level = set([])
    for doors in [left_doors, right_doors, upper_doors, lower_doors]:
        for (xi, yi) in zip(x, y):
            if (xi, yi) in doors:
                is_door_complete = True
                for xj, yj in doors:
                    is_door_complete = is_door_complete and (xj in x) and (yj in y)

                if is_door_complete:
                    doors_present_in_level.add(tuple(doors))

    return doors_present_in_level


def has_doors_or_stairs(
    level: np.ndarray, possible_door_positions, num_doors: int = 1
) -> bool:
    """
    Checks if it has doors in the right places
    """
    stairs_x, _ = np.where(level == 10)
    if len(stairs_x) > 0:
        # there are stairs
        return True

    x, y = np.where(level == 3)
    if len(x) == 0:
        # There are no doors :(
        return False

    # Check if the doors are in sensible positions
    flag_ = True
    for xi, yi in zip(x, y):
        flag_ = flag_ and ((xi, yi) in possible_door_positions)

    # Making sure doors are complete
    left_doors = [(4, 1), (5, 1), (6, 1)]
    right_doors = [(4, 14), (5, 14), (6, 14)]
    upper_doors = [(1, 7), (1, 8)]
    lower_doors = [(9, 7), (9, 8)]
    doors_present_in_level = set([])
    for doors in [left_doors, right_doors, upper_doors, lower_doors]:
        for (xi, yi) in zip(x, y):
            if (xi, yi) in doors:
                doors_present_in_level.add(tuple(doors))
                for xj, yj in doors:
                    flag_ = flag_ and (xj in x) and (yj in y)

    return flag_ and (len(doors_present_in_level) >= num_doors)


def get_neighbors(position: Tuple[int, int]):
    """
    Given (i, j) in {position}, returns
    all 8 neighbors (i-1, j-1), ..., (i+1, j+1).
    """
    i, j = position
    width, height = 16, 11

    if i < 0 or i >= height:
        raise ValueError(f"Position is out of bounds in x: {position}")

    if j < 0 or j >= width:
        raise ValueError(f"Position is out of bounds in x: {position}")

    neighbors = []

    if i - 1 >= 0:
        if j - 1 >= 0:
            neighbors.append((i - 1, j - 1))

        if j + 1 < width:
            neighbors.append((i - 1, j + 1))

        neighbors.append((i - 1, j))

    if i + 1 < height:
        if j - 1 >= 0:
            neighbors.append((i + 1, j - 1))

        if j + 1 < width:
            neighbors.append((i + 1, j + 1))

        neighbors.append((i + 1, j))

    if j - 1 >= 0:
        neighbors.append((i, j - 1))

    if j + 1 < width:
        neighbors.append((i, j + 1))

    random.shuffle(neighbors)

    return neighbors


def are_doors_connected(level, door_1, door_2):
    doors_in_level = get_doors_in_level(level)
    assert door_1 in doors_in_level
    assert door_2 in doors_in_level

    if len(doors_in_level) <= 1:
        return True

    passable_blocks = [encoding[s] for s in ["F", "D", "O", "-"]]
    first_position = list(door_1)[0]

    neighbors = get_neighbors(first_position)
    q = Queue()
    for n in neighbors:
        q.put(n)

    visited_positions = set([first_position])
    while not q.empty():
        v = q.get()

        if v in visited_positions:
            continue
        visited_positions.add(v)

        if v in door_2:
            return True

        for n in get_neighbors(v):
            if level[n] in passable_blocks:
                if n not in visited_positions:
                    q.put(n)

    return False


if __name__ == "__main__":
    onehot = np.load("./data/processed/zelda/onehot.npz")["levels"]
    levels = onehot.argmax(axis=-1)

    print(levels)
    b, x, y = np.where(levels == 3)
    print(np.where(levels == 3))
    for i, level in enumerate(levels):
        # print(level)
        if not grammar_check(level):
            print(level)

    print(len([l for l in levels if not grammar_check(l)]))
