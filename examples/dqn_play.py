#!/usr/bin/env python3
###############################################################################################################
# dqn_play.py – Play MinAtar Breakout with a trained DQN agent and save a GIF of gameplay
#
# Example:
#   python dqn_play.py -g breakout -m breakout_data_and_weights --gif breakout.gif
#
#   generates 10 episodes of gameplay using model saved in breakout_data_and_weights and saves a GIF to 
#   breakout.gif with 0.5s between frames
###############################################################################################################
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import imageio.v2 as imageio

from minatar import Environment


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Network identical to the one used during training
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def after_conv(size, k=3, s=1):
            return (size - (k - 1) - 1) // s + 1

        fc_in = after_conv(10) * after_conv(10) * 16
        self.fc_hidden = nn.Linear(fc_in, 128)
        self.output = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)


def to_tensor(state_np):
    """10x10xC numpy → (1,C,10,10) float32 tensor on DEVICE"""
    return (
        torch.tensor(state_np, device=DEVICE)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Frame builder with cubehelix palette
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_palette(n_channels: int) -> np.ndarray:
    """
    Returns (n_channels+1, 3) uint8 palette where index 0 is black and
    indices 1…n correspond to seaborn's cubehelix colours.
    """
    pal = sns.color_palette("cubehelix", n_channels)
    pal = [(0, 0, 0)] + [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in pal]
    return np.asarray(pal, dtype=np.uint8)


def state_to_rgb(state: np.ndarray, palette: np.ndarray, scale: int = 40) -> np.ndarray:
    """
    Map MinAtar binary state (10×10×C) to an up-scaled RGB image using
    the provided palette.
    """
    has_obj = state.any(axis=2)
    idx = state.argmax(axis=2) + 1       # 1…C
    idx[~has_obj] = 0                    # background
    rgb_small = palette[idx]             # 10×10×3 uint8
    return np.repeat(np.repeat(rgb_small, scale, axis=0), scale, axis=1)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Play & record
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def play(game, model_path, episodes, pause, gif_path):
    env = Environment(game)
    n_channels, num_actions = env.state_shape()[2], env.num_actions()

    # cubehelix palette
    palette = build_palette(n_channels)

    # Load network
    policy_net = QNetwork(n_channels, num_actions).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    policy_net.load_state_dict(ckpt["policy_net_state_dict"])
    policy_net.eval()

    # GIF writer (optional)
    gif_writer = None
    if gif_path:
        gif_writer = imageio.get_writer(
            gif_path,
            mode="I",
            duration=pause,
            loop=0,
        )
        print(f"Recording episodes to '{gif_path}'")

    returns = []
    for ep in range(1, episodes + 1):
        G, terminated = 0.0, False
        env.reset()
        state = env.state()

        while not terminated:
            env.display_state(pause)

            if gif_writer:
                frame = state_to_rgb(state, palette)
                gif_writer.append_data(frame)

            with torch.no_grad():
                action = policy_net(to_tensor(state)).argmax(1).item()
            reward, terminated = env.act(action)
            state = env.state()
            G += reward

        returns.append(G)
        print(f"Episode {ep:>3}/{episodes}: return = {G}")

    env.close_display()
    if gif_writer:
        gif_writer.close()
        print(f"GIF saved: {gif_path}")

    print("-" * 60)
    print(
        f"Average return over {episodes} episodes: "
        f"{np.mean(returns):.2f} ± {np.std(returns) / np.sqrt(episodes):.2f}"
    )


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", required=True, help="MinAtar game (e.g. 'breakout')")
    parser.add_argument(
        "-m",
        "--model",
        default="breakout_data_and_weights",
        help="Path to the [args.game]_data_and_weights file produced by training",
    )
    parser.add_argument("-n", "--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--delay", type=int, default=0.5, help="Delay between frames in s (for display & GIF)")
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="Filename for GIF output; omit to disable recording",
    )
    args = parser.parse_args()

    gif_file = Path(args.gif).expanduser().resolve() if args.gif else None
    play(args.game, args.model, args.episodes, args.delay, str(gif_file) if gif_file else None)
