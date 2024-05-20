"""Script to generate the activations datasets.

Run with:
```bash
poetry run python -m scripts.datasets.make_activations_dataset
```
"""

import re
import argparse
from dataclasses import dataclass
import copy

import chess
from datasets import load_dataset, DatasetDict, Dataset
from loguru import logger
from lczerolens import ModelWrapper
from torch.utils.data import DataLoader
from lczerolens.xai.hook import CacheHook, HookConfig
import torch

from scripts.constants import HF_TOKEN, SMALL_MODELS, MEDIUM_MODELS, BIG_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch_gen(wdls, acts, infos):
    prev_root = None
    prev_opt = None
    for wdl, act, info in zip(wdls, acts, infos):
        is_root = info.pop("is_root")
        if is_root:
            prev_root = (wdl, act, info)
            prev_opt = None
            continue
        if "opt_fen" in info:
            prev_opt = (wdl, act, info)
            continue
        root_wdl, root_act, root_info = prev_root
        opt_wdl, opt_act, opt_info = prev_opt
        info["root_fen"] = root_info["root_fen"]
        info["opt_fen"] = opt_info["opt_fen"]
        info["root_act"] = root_act
        info["opt_act"] = opt_act
        info["sub_act"] = act
        info["root_wdl"] = root_wdl
        info["opt_wdl"] = opt_wdl
        info["sub_wdl"] = wdl
        prev_opt = None
        yield info


@torch.no_grad
def gen_fn(dataloader, wrapper, cache_hook, layer):
    module_exp = re.compile(rf".*block{layer}/conv2/relu")
    for batch in dataloader:
        boards, infos = batch
        stats = wrapper.predict(boards)[0]
        for module, batched_activations in cache_hook.storage.items():
            m = module_exp.match(module)
            if m is None:
                raise ValueError(f"Invalid module: {module}")
            yield from make_batch_gen(
                stats["wdl"].detach().cpu().float().numpy(), batched_activations.detach().cpu().float().numpy(), infos
            )


def param_collate_fn(batch, min_depth):
    boards, infos = [], []
    for x in batch:
        fen = x["fen"]
        moves_opt = x["moves_opt"]
        moves_sub = x["moves_sub0"]
        board = chess.Board(fen)
        root_fen = None
        max_depth = min(len(moves_opt), len(moves_sub))
        if max_depth < min_depth + 7:
            continue
        for i in range(7):
            board.push(chess.Move.from_uci(moves_opt[i]))
        root_fen = board.fen()
        x["root_fen"] = root_fen
        x["is_root"] = True
        boards.append(board.copy(stack=7))
        infos.append(copy.deepcopy(x))
        board_opt = board.copy(stack=7)
        board_sub = board.copy(stack=7)
        for i, (move_opt, move_sub) in enumerate(zip(moves_opt[7:], moves_sub[7:])):
            board_opt.push(chess.Move.from_uci(move_opt))
            board_sub.push(chess.Move.from_uci(move_sub))
            if i + 1 >= min_depth:
                x["is_root"] = False
                x["current_depth"] = i + 1
                boards.append(board_opt.copy())
                infos.append({**copy.deepcopy(x), "opt_fen": board_opt.fen()})
                boards.append(board_sub.copy())
                infos.append({**copy.deepcopy(x), "sub_fen": board_sub.fen()})

    return boards, infos


def main(args: argparse.Namespace):
    logger.info(f"Loading `{args.source_dataset}`...")
    dataset = load_dataset(args.source_dataset, args.source_config)

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
        config_name = f"{args.source_config}_{model}"
        wrapper = ModelWrapper.from_onnx_path(f"./assets/models/{model}").to(DEVICE)
        filtered_dataset = dataset.filter(
            lambda s: s["depth_opt"] >= args.min_depth and s["depth_sub0"] >= args.min_depth
        )
        if args.debug:
            filtered_dataset = DatasetDict(
                {
                    "train": filtered_dataset["train"].select(range(3000)),
                    "test": filtered_dataset["test"].select(range(500)),
                }
            )
        cache_hook = CacheHook(HookConfig(module_exp=rf".*block{args.layer}/conv2/relu"))
        cache_hook.register(wrapper)

        _dataset_dict = {}
        for split in ["train", "test"]:

            def collate_fn(batch):
                return param_collate_fn(batch, args.min_depth)

            dataloader = DataLoader(
                filtered_dataset[split],
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            )
            _dataset_dict[split] = Dataset.from_generator(
                gen_fn,
                gen_kwargs={
                    "dataloader": dataloader,
                    "wrapper": wrapper,
                    "cache_hook": cache_hook,
                    "layer": args.layer,
                },
            )

        dataset_dict = DatasetDict(_dataset_dict)
        logger.info(f"Processed dataset: {dataset_dict}")
        if args.push_to_hub:
            logger.info("Pushing to hub...")
            dataset_dict.push_to_hub(
                repo_id=args.dataset_name,
                token=HF_TOKEN,
                config_name="debug" if args.debug else f"{config_name}_{args.layer}",
            )


@dataclass
class Args:
    source_dataset: str = "Xmaster6y/lczero-planning-trajectories"
    source_config: str = "lc0-10-4238.onnx-policy"
    dataset_name: str = "Xmaster6y/lczero-planning-activations"
    push_to_hub: bool = False
    model_category: str = "small"
    use_all_models: bool = False
    layer: str = "9"
    min_depth: int = 10
    batch_size: int = 1000
    debug: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-activations-dataset")
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="Xmaster6y/lczero-planning-trajectories",
    )
    parser.add_argument(
        "--source_config",
        type=str,
        default="lc0-10-4238.onnx-policy",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Xmaster6y/lczero-planning-activations",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model_category", type=str, default="small")
    parser.add_argument("--use_all_models", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer", type=str, default="9")
    parser.add_argument("--min_depth", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
