import matplotlib.pyplot as plt
import numpy as np

def translate_standard_features(features):
    return {
        "awareness_up": features[0],
        "awareness_right": features[1],
        "awareness_down": features[2],
        "awareness_left": features[3],
        "danger_current": features[4],
        "danger_up": features[5],
        "danger_right": features[6],
        "danger_down": features[7],
        "danger_left": features[8],
        "safe_direction_none": features[9],
        "safe_direction_up": features[10],
        "safe_direction_right": features[11],
        "safe_direction_down": features[12],
        "safe_direction_left": features[13],
        "coin_direction_up": features[14],
        "coin_direction_right": features[15],
        "coin_direction_down": features[16],
        "coin_direction_left": features[17],
        "crate_direction_current": features[18],
        "crate_direction_up": features[19],
        "crate_direction_right": features[20],
        "crate_direction_down": features[21],
        "crate_direction_left": features[22],
        "bomb_info_bomb_possible": features[23],
        "bomb_info_survival_possible": features[24]
    }


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

    channels = np.stack([game_map, coin_map, self_map])
    channels = np.moveaxis(channels, 0, -1)

    channels = channels.astype(np.float)
    plt.imshow(channels[:, :, :])
    plt.show()

