"""Matchmaking to determine sampling hyperparameters.

Run with:
```bash
poetry run python -m scripts.play.raw_matchmaking
```
"""

import argparse
import random

from loguru import logger
from lczerolens import ModelWrapper
from lczerolens.game import SelfPlay, WrapperSampler, PolicySampler
import chess
import wandb
import torch

from scripts.constants import SMALL_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALUES = [0, 0.25, 0.5, 0.75, 1.0]


def generic_report_fn(to_log, player, wandb_run):
    color = "white" if player == chess.WHITE else "black"
    wandb_run.log({f"{color}/{k}": v for k, v in to_log.items()})


def main(args: argparse.Namespace):
    logger.info(f"Running matchmaking with {args.n_games} games and {args.max_moves} moves...")
    logger.info(f"Models: {SMALL_MODELS}")
    for model in SMALL_MODELS:
        wrapper = ModelWrapper.from_onnx_path(f"./assets/{model}").to(DEVICE)
        for _ in range(args.n_games):
            for contender_color in [chess.WHITE, chess.BLACK]:
                alpha = random.choice(VALUES)
                beta = random.choice(VALUES)
                gamma = random.choice(VALUES)
                contender = WrapperSampler(
                    wrapper,
                    use_argmax=args.use_argmax,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                policy = PolicySampler(
                    wrapper,
                    use_argmax=args.use_argmax,
                )
                if contender_color == chess.WHITE:
                    white = contender
                    black = policy
                    color = "white"
                else:
                    white = policy
                    black = contender
                    color = "black"

                logger.info(f"Model: {model}")
                logger.info(f"Contender params: {alpha}, {beta}, {gamma}")
                with wandb.init(
                    entity="yp-edu",
                    project="raw-matchmaking",
                    config={
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "contender_color": color,
                        "model": model,
                    },
                ) as wandb_run:

                    def report_fn(to_log, player):
                        generic_report_fn(to_log, player, wandb_run)

                    matchmaking = SelfPlay(
                        white,
                        black,
                    )
                    game, board = matchmaking.play(report_fn=report_fn, max_moves=args.max_moves)
                    result = board.result()
                    logger.info(f"Game length: {len(game)}")
                    logger.info(f"Final result: {result}")
                    if result == "1-0":
                        white_reward = 1
                        black_reward = -1
                    elif result == "0-1":
                        white_reward = -1
                        black_reward = 1
                    else:
                        white_reward = 0
                        black_reward = 0
                    contender_reward = white_reward if contender_color == chess.WHITE else black_reward
                    wandb_run.log(
                        {
                            f"{color}/reward": contender_reward,
                            "contender_reward": contender_reward,
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("raw-matchmaking")
    parser.add_argument("--n_games", type=int, default=30)
    parser.add_argument("--use_argmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_moves", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
