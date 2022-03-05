
import numpy as np
import matplotlib.pyplot as plt
from items import Coin
from debug_utils import print_field
from agent_code.n_step_agent_v2.callbacks import get_coin_direction, get_bomb_map


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

game_state = {
    "round": 1,
    "step": 1,
    "field": arena[0],
    "coins": arena[1],
    "self": ("my_agent", 0, True, (1, 1)),
    "bombs": [((3, 3), 4)],
    "explosion_map": np.zeros_like(arena[0])
}

print_field(game_state=game_state)


coin_direction = get_coin_direction(game_state["field"], game_state["coins"], game_state["self"][3])


bomb_map = get_bomb_map(game_state["field"], game_state["bombs"], game_state["explosion_map"])
plt.imshow(bomb_map)
plt.show()





