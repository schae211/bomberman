
import numpy as np
import matplotlib.pyplot as plt
from items import Coin
from debug_utils import *
from agent_code.nn_agent_v2.callbacks import *


def build_arena(COLS, ROWS, CRATE_DENSITY, COIN_COUNT, SEED):
    WALL = -1
    FREE = 0
    CRATE = 1
    arena = np.zeros((COLS, ROWS), int)

    rng = np.random.default_rng(SEED)
    # Crates in random locations
    arena[rng.random((COLS, ROWS)) < CRATE_DENSITY] = CRATE

    # Walls
    arena[:1, :] = WALL
    arena[-1:, :] = WALL
    arena[:, :1] = WALL
    arena[:, -1:] = WALL
    for x in range(COLS):
        for y in range(ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = WALL

    # Clean the start positions
    start_positions = [(1, 1), (1, ROWS - 2), (COLS - 2, 1), (COLS - 2, ROWS - 2)]
    for (x, y) in start_positions:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if arena[xx, yy] == 1:
                arena[xx, yy] = FREE

    # Place coins at random, at preference under crates
    coins = []
    all_positions = np.stack(np.meshgrid(np.arange(COLS), np.arange(ROWS), indexing="ij"), -1)
    crate_positions = rng.permutation(all_positions[arena == CRATE])
    free_positions = rng.permutation(all_positions[arena == FREE])
    coin_positions = np.concatenate([
        crate_positions,
        free_positions
    ], 0)[:COIN_COUNT]
    for x, y in coin_positions:
        coins.append(Coin((x, y), collectable=arena[x, y] == FREE))

    collectable_coins = [(coin.x, coin.y) for coin in coins if coin.collectable]

    return arena, collectable_coins

scenario = "enemy_close_by"

if scenario == "1":
    arena = build_arena(17, 17, 0.2, 50, 42)
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 1)),
        "others": [],
        "bombs": [((3, 3), 3), ((1, 1), 0)],
        "explosion_map": np.zeros_like(arena[0])
    }
elif scenario == "2":
    arena = build_arena(17, 17, 0.2, 50, 42)
    arena[0][2, 1] = -1
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 2)),
        "others": [],
        "bombs": [((1, 3), 3), ((4, 5), 3)],
        "explosion_map": np.zeros_like(arena[0])
    }

# testing if bfs check that bomb danger is already gone
elif scenario == "3":
    arena = build_arena(17, 17, 0.2, 50, 42)
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 2)),
        "others": [],
        "bombs": [((1, 3), 3), ((2, 3), 3), ((4, 1), 0)],
        "explosion_map": np.zeros_like(arena[0])
    }

elif scenario == "suicide":
    arena = build_arena(17, 17, 0.7, 50, 42)
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 1)),
        "others": [],
        "bombs": [],
        "explosion_map": np.zeros_like(arena[0])
    }

elif scenario == "crate_far_away":
    arena = build_arena(17, 17, 0.7, 50, 42)
    field = arena[0]
    for x in range(0, 8):
        for y in range(0, 8):
            if field[x, y] == 1:
                field[x, y] = 0
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 1)),
        "others": [],
        "bombs": [],
        "explosion_map": np.zeros_like(arena[0])
    }

elif scenario == "enemy_close_by":
    arena = build_arena(17, 17, 0.7, 50, 42)
    field = arena[0]
    for x in range(0, 8):
        for y in range(0, 8):
            if field[x, y] == 1:
                field[x, y] = 0
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 1)),
        "others": [("other_agent", 0, True, (3, 2))],
        "bombs": [],
        "explosion_map": np.zeros_like(arena[0])
    }


object_map = game_state["field"]

coin_map = np.zeros_like(game_state["field"])
for cx, cy in game_state["coins"]: coin_map[cx, cy] = 1

self_map = np.zeros_like(game_state["field"])
self_map[game_state["self"][3]] = 1

explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                             explosion_position=game_state["explosion_map"])

other_agents = np.zeros_like(game_state["field"])
for (_, _, _, (cx, cy)) in game_state["others"]: other_agents[cx, cy] = 1

# create channels based on the field and coin information.
channels = [object_map, self_map, coin_map, explosion_map, other_agents]

# concatenate them as a feature tensor
stacked_channels = np.stack(channels)


# 1. 1D array, len = 4: indicating in which directions the agent can move (up, right, down, left)
awareness = get_awareness(object_position=game_state["field"], self_position=game_state["self"][3])

# 2. 1D array, len = 4:  indicating in which direction lays the closest coin determined by BFS (up, right, down, left)
coin_direction = get_coin_direction(object_position=game_state["field"], coin_list=game_state["coins"],
                                    self_position=game_state["self"][3])

# 2D array indicating which area is affected by exploded bombs or by bombs which are about to explode
explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                             explosion_position=game_state["explosion_map"])

# 3. 1D array, len = 5:  indicating how dangerous the current field, up, right, down, left are
danger = get_danger(explosion_map=explosion_map, self_position=game_state["self"][3])

# 4. 1D array, len = 5:  indicating in which direction to flee from bomb if immediate danger (current, up, right, down, left),
# if no immediate danger/current spot is safe returns all zeros, otherwise direction
safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                    self_position=game_state["self"][3])

# 5. 1D array, len = 5:  indicating in which direction lays the most lucrative position for laying a bomb
# that destroys most crates penalized by the distance as determined by BFS (up, right, down, left)
crate_direction = get_crate_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                      self_position=game_state["self"][3], explosion_map=explosion_map)

# 6. 1D array, len = 2: indicating whether bomb can be dropped and survival is possible
bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                          self=game_state["self"], bomb_list=game_state["bombs"])

# TODO: Add other agents information, to aid attacking them
#   Add feature vector with len 5 indicating whether target was acquired and in which direction
others_direction = get_others_direction(object_position=game_state["field"], bomb_list=game_state["bombs"],
                                        self_position=game_state["self"][3], explosion_map=explosion_map,
                                        others=game_state["others"])

features = np.concatenate([awareness,
                           danger,
                           safe_direction,
                           coin_direction,
                           crate_direction,
                           bomb_info,
                           others_direction])

print_field(game_state)
print_channels(channels)



