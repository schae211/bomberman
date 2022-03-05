
import numpy as np
import matplotlib.pyplot as plt
from items import Coin
from debug_utils import *
from agent_code.n_step_agent_v2.callbacks import *


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


arena = build_arena(17, 17, 0.2, 50, 42)

scenario = "2"

if scenario == "1":
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
    game_state = {
        "round": 1,
        "step": 1,
        "field": arena[0],
        "coins": arena[1],
        "self": ("my_agent", 0, True, (1, 1)),
        "bombs": [((1, 3), 3)],
        "explosion_map": np.zeros_like(arena[0])
    }

print_field(game_state=game_state)

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
plt.imshow(explosion_map)
plt.show()
save_info = save_bfs(game_state["field"], explosion_map, game_state["self"][3])

# 1D array indicating in which direction to flee from bomb if immediate danger
# if no immediate danger return 5 zeros or current spot is safe returns all zeros, otherwise direction
safe_direction = get_safe_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                    self_position=game_state["self"][3])

# 1D array indicating whether current field, up, right, down, left are dangerous
danger = get_danger(explosion_map=explosion_map, self_position=game_state["self"][3])

crate_direction = get_crate_direction(object_position=game_state["field"], explosion_map=explosion_map,
                                      self_position=game_state["self"][3])
crate_info = crate_bfs(game_state["field"], game_state["self"][3], explosion_map)

# 1D array indicating whether bomb can be dropped and survival is possible
bomb_info = get_bomb_info(object_position=game_state["field"], explosion_map=explosion_map,
                          self=game_state["self"])

features = np.concatenate([awareness,
                           danger,
                           safe_direction,
                           coin_direction,
                           crate_direction,
                           bomb_info])





