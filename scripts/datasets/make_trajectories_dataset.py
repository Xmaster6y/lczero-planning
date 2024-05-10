"""Script to generate the trajectories datasets.

Run with:
```bash
poetry run python -m scripts.datasets.make_trajectories_dataset
```
"""

import argparse
from dataclasses import dataclass
from typing import List

import chess
from datasets import load_dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError
from loguru import logger
from lczerolens import ModelWrapper
from lczerolens import move_encodings
import torch
from torch.distributions import Categorical

from scripts.constants import HF_TOKEN, SMALL_MODELS, MEDIUM_MODELS, BIG_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BatchedPolicySampler:
    wrapper: ModelWrapper
    use_argmax: bool = True
    use_suboptimal: bool = False

    @torch.no_grad
    def get_next_moves(
        self,
        boards: List[chess.Board],
    ):
        all_stats = self.wrapper.predict(boards)[0]
        if isinstance(all_stats, list):
            batched_policy = all_stats[0]
        else:
            batched_policy = all_stats["policy"]
        for board, policy in zip(boards, batched_policy):
            us = board.turn
            indices = torch.tensor([move_encodings.encode_move(move, (us, not us)) for move in board.legal_moves]).to(
                DEVICE
            )
            legal_policy = policy.gather(0, indices)
            if self.use_argmax:
                idx = legal_policy.argmax()
            else:
                if self.use_suboptimal:
                    idx = legal_policy.argmax()
                    legal_policy[idx] = torch.tensor(-1e3)
                m = Categorical(logits=legal_policy)
                idx = m.sample()
            yield list(board.legal_moves)[idx]


def make_samplers(
    model: str,
):
    wrapper = ModelWrapper.from_onnx_path(f"./assets/models/{model}").to(DEVICE)
    optimal_sampler = BatchedPolicySampler(
        wrapper=wrapper,
        use_argmax=True,
    )
    suboptimal_sampler = BatchedPolicySampler(
        wrapper=wrapper,
        use_argmax=False,
    )
    return optimal_sampler, suboptimal_sampler


def batched_sample_trajectory(
    batch,
    sampler: BatchedPolicySampler,
    max_depth: int,
    suffix: str,
):
    output_batch = {"gameid": [], "fen": [], f"moves_{suffix}": [], f"depth_{suffix}": []}
    boards = []
    for fen, moves in zip(batch["fen"], batch["moves"]):
        board = chess.Board(fen)
        for move in moves:
            board.push(chess.Move.from_uci(move))
        boards.append(board)

    depth = 0
    all_moves = batch["moves"]
    gameids = batch["gameid"]
    fens = batch["fen"]
    while depth < max_depth and len(boards) > 0:
        working_boards = []
        working_moves = []
        working_gameids = []
        working_fens = []
        for i, board in enumerate(boards):
            if board.is_game_over() or (depth > 4 and i == 2):
                output_batch["gameid"].append(gameids[i])
                output_batch["fen"].append(fens[i])
                output_batch[f"moves_{suffix}"].append(all_moves[i])
                output_batch[f"depth_{suffix}"].append(depth)
            else:
                working_boards.append(board)
                working_moves.append(all_moves[i].copy())
                working_gameids.append(gameids[i])
                working_fens.append(fens[i])
        if len(working_boards) == 0:
            break
        next_moves = sampler.get_next_moves(working_boards)
        for i, (board, move) in enumerate(zip(working_boards, next_moves)):
            board.push(move)
            working_moves[i].append(move.uci())
        depth += 1
        boards = working_boards
        all_moves = working_moves
        gameids = working_gameids
        fens = working_fens
    for i in range(len(boards)):
        output_batch["gameid"].append(gameids[i])
        output_batch["fen"].append(fens[i])
        output_batch[f"moves_{suffix}"].append(all_moves[i])
        output_batch[f"depth_{suffix}"].append(depth)
    return output_batch


def map_fn(
    batch,
    samplers,
    max_depth: int,
    suboptimal_resample: int,
):
    otpimal_sampler, suboptimal_sampler = samplers
    output_batch = batched_sample_trajectory(batch, otpimal_sampler, max_depth, "opt")
    for i in range(suboptimal_resample):
        suboptimal_batch = batched_sample_trajectory(batch, suboptimal_sampler, max_depth, f"sub{i}")
        output_batch[f"moves_sub{i}"] = suboptimal_batch[f"moves_sub{i}"]
        output_batch[f"depth_sub{i}"] = suboptimal_batch[f"depth_sub{i}"]
    return output_batch


def main(args: argparse.Namespace):
    logger.info(f"Loading `{args.source_dataset}`...")
    dataset = load_dataset(args.source_dataset)

    if args.model_category == "small":
        selected_models = SMALL_MODELS
    elif args.model_category == "medium":
        selected_models = MEDIUM_MODELS
    elif args.model_category == "big":
        selected_models = BIG_MODELS
    else:
        raise ValueError("Invalid model category")
    if not args.use_all_models:
        selected_models = [selected_models[0]]
    logger.info(f"Selected models: {selected_models}")

    for model in selected_models:
        if args.use_policy:
            config_name = f"{model}-policy"
        else:
            raise NotImplementedError("Only policy sampling is supported")
        dataset_dict = DatasetDict()
        if not args.restart:
            try:
                continue_dataset = load_dataset(args.dataset_name, name=config_name)
            except DatasetNotFoundError:
                continue_dataset = None
        else:
            continue_dataset = None
        for split, n_samples in [("train", args.train_samples), ("test", args.test_samples)]:
            start_from = 0 if continue_dataset is None else len(continue_dataset[split])
            logger.info(f"Starting from {start_from}...")
            split_dataset = (
                dataset[split]
                .select(range(start_from, start_from + n_samples))
                .map(
                    map_fn,
                    fn_kwargs={
                        "samplers": make_samplers(model),
                        "max_depth": args.max_depth,
                        "suboptimal_resample": args.suboptimal_resample,
                    },
                    batched=True,
                    batch_size=args.batch_size,
                    num_proc=args.num_proc,
                )
            )
            if continue_dataset is not None:
                split_dataset = continue_dataset[split].concatenate(split_dataset)
            dataset_dict[split] = split_dataset

        logger.info(f"Processed dataset: {dataset_dict}")

        if args.push_to_hub:
            logger.info("Pushing to hub...")
            dataset_dict.push_to_hub(
                repo_id=args.dataset_name,
                token=HF_TOKEN,
                config_name=config_name,
            )


@dataclass
class Args:
    source_dataset: str = "Xmaster6y/lczero-planning-boards"
    dataset_name: str = "Xmaster6y/lczero-planning-trajectories"
    push_to_hub: bool = False
    use_policy: bool = True
    model_category: str = "small"
    use_all_models: bool = False
    train_samples: int = 100_000
    test_samples: int = 10_000
    restart: bool = False
    max_depth: int = 15
    suboptimal_resample: int = 1
    batch_size: int = 1000
    num_proc: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-trajectories-dataset")
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="Xmaster6y/lczero-planning-boards",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Xmaster6y/lczero-planning-trajectories",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_category", type=str, default="small")
    parser.add_argument("--use_all_models", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_samples", type=int, default=100_000)
    parser.add_argument("--test_samples", type=int, default=10_000)
    parser.add_argument("--restart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--suboptimal_resample", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_proc", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
