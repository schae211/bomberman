
import numpy as np
import matplotlib.pyplot as plt
from items import Coin
from debug_utils import *
from agent_code.nn_agent_v1.callbacks import *


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

scenario = "suicide"

if scenario == "1":
    arena = build_arena(17, 17, 0.2, 50, 42)
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 1)),
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
        "bombs": [((1, 3), 4), ((4, 5), 4)],
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
        "bombs": [((1, 3), 4), ((2, 3), 4), ((4, 1), 0)],
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
        "bombs": [],
        "explosion_map": np.zeros_like(arena[0])
    }

# Situational awareness, indicating in which directions the agent can move
# Add is bomb action possible?
awareness = get_awareness(object_position=game_state["field"], self_position=game_state["self"][3])

# Direction to the closest coin determined by BFS
coin_direction = get_coin_direction(object_position=game_state["field"], coin_list=game_state["coins"],
                                    self_position=game_state["self"][3])
coin_info = coin_bfs(game_state["field"], game_state["coins"], game_state["self"][3])

# 2D array indicating which area is affected by exploded bombs or by bombs which are about to explode
explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                             explosion_position=game_state["explosion_map"])
save_info = save_bfs(game_state["field"], explosion_map, game_state["self"][3])

# 1D array indicating in which direction to flee from bomb if immediate danger
# if no immediate danger return 5 zeros or current spot is safe returns all zeros, otherwise direction
safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                    self_position=game_state["self"][3])

# 1D array indicating whether current field, up, right, down, left are dangerous
danger = get_danger(explosion_map=explosion_map, self_position=game_state["self"][3])

crate_direction = get_crate_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                      self_position=game_state["self"][3], bomb_list=game_state["bombs"])
crate_info = crate_bfs(object_position=game_state["field"], self_position=game_state["self"][3],
                       explosion_map=explosion_map, bomb_list=game_state["bombs"])

# 1D array indicating whether bomb can be dropped and survival is possible
bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                          self=game_state["self"], bomb_list=game_state["bombs"])

features = np.concatenate([awareness,
                           danger,
                           safe_direction,
                           coin_direction,
                           crate_direction,
                           bomb_info])

# channels
object_map = game_state["field"]

# for the coin challenge we need to know where the agent is, where walls are and where the coins are
# so, we create a coin map in the same shape as the field
coin_map = np.zeros_like(game_state["field"])
for cx, cy in game_state["coins"]:
    coin_map[cx, cy] = 1

# also adding where one self is on the map
self_map = np.zeros_like(game_state["field"])
self_map[game_state["self"][3]] = 1

explosion_map = get_bomb_map(object_position=game_state["field"], bomb_list=game_state["bombs"],
                             explosion_position=game_state["explosion_map"])

# create channels based on the field and coin information.
channels = [object_map, self_map, coin_map, explosion_map]

# concatenate them as a feature tensor (they must have the same shape), ...
stacked_channels = np.stack(channels)
# and return them as a vector
stacked_channels = stacked_channels.reshape(-1)

print_channels(channels)



