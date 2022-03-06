import matplotlib.pyplot as plt
import numpy as np

feature_description = [
    "awareness_up",
    "awareness_right",
    "awareness_down",
    "awareness_left",
    "danger_current",
    "danger_up",
    "danger_right",
    "danger_down",
    "danger_left",
    "safe_direction_none",
    "safe_direction_up",
    "safe_direction_right",
    "safe_direction_down",
    "safe_direction_left",
    "coin_direction_up",
    "coin_direction_right",
    "coin_direction_down",
    "coin_direction_left",
    "crate_direction_current",
    "crate_direction_up",
    "crate_direction_right",
    "crate_direction_down",
    "crate_direction_left",
    "bomb_info_bomb_possible",
    "bomb_info_survival_possible"
]


def print_field(game_state):
    # Red
    game_map = game_state["field"].copy()
    game_map_1 = np.where(game_map == 0, 0.8, 0)
    game_map_2 = np.where(game_map == 1, 0.4, 0)
    game_map = game_map_1 + game_map_2

    # Green
    coin_map = np.zeros_like(game_state["field"], dtype=float)
    for cx, cy in game_state["coins"]:
        coin_map[cx, cy] = 1

    # Blue
    self_map = np.zeros_like(game_state["field"], dtype=float)
    self_map[game_state["self"][3]] = 1

    # add bomb map to coin map
    bomb_map = np.zeros_like(game_state["field"], dtype=float)
    for (x, y), _ in game_state["bombs"]:
        bomb_map[x, y] = 0.5

    game_map += bomb_map
    coin_map += bomb_map

    channels = np.stack([game_map, coin_map, self_map])
    channels = np.moveaxis(channels, 0, -1)

    channels = channels.astype(np.float)
    plt.imshow(channels[:, :, :])
    plt.show()


def print_channels(channels):
    object_map, self_map, coin_map, explosion_map = channels[0], channels[1], channels[2], channels[3]

    fig, axis = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    axis = axis.flatten()
    im1 = axis[0].imshow(object_map)
    axis[0].set_title("Object Map")
    plt.colorbar(im1, ax=axis[0])

    im2 = axis[1].imshow(coin_map)
    axis[1].set_title("Coin Map")
    plt.colorbar(im2, ax=axis[1])

    im3 = axis[2].imshow(explosion_map)
    axis[2].set_title("Explosion Map")
    plt.colorbar(im3, ax=axis[2])

    im4 = axis[3].imshow(self_map)
    axis[3].set_title("Self Map")
    plt.colorbar(im4, ax=axis[3])
    plt.show()


