"""Script to generate the activations datasets.

Run with:
```bash
poetry run python -m scripts.datasets.make_activations_dataset
```
"""

import re
import argparse
from dataclasses import dataclass

import chess
from datasets import load_dataset, DatasetDict, Dataset
from loguru import logger
from lczerolens import ModelWrapper
from torch.utils.data import DataLoader
from lczerolens.xai.hook import CacheHook, HookConfig
import torch

from scripts.constants import HF_TOKEN, SMALL_MODELS, MEDIUM_MODELS, BIG_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def merge_gens(gens):
    def new_gen():
        for gen in gens:
            yield from gen()

    return new_gen


def make_batch_gen(batch, infos, out_name):
    def gen():
        for tensor, info in zip(batch, infos):
            yield {out_name: tensor.cpu().float().numpy(), **info}

    return gen


@torch.no_grad
def make_gen_list(gen_dict, dataloaders, wrapper, cache_hook):
    module_exp = re.compile(r".*block(?P<layer>\d+)$")
    for split, dataloader in dataloaders.items():
        for batch in dataloader:
            boards, infos = batch
            wrapper.predict(boards)
            logger.info(f"Processed batch of size {len(boards)}")
            for module, batched_activations in cache_hook.storage.items():
                m = module_exp.match(module)
                layer = m.group("layer")
                if layer not in gen_dict:
                    gen_dict[layer] = {"test": [], "train": []}
                gen_dict[layer][split].append(make_batch_gen(batched_activations[0].detach(), infos, "activation"))


def param_collate_fn(batch, min_depth):
    boards, infos = [], []
    for x in batch:
        fen = x["fen"]
        moves = x["moves"]
        board = chess.Board(fen)
        root_fen = None
        for i, move in enumerate(moves):
            board.push(chess.Move.from_uci(move))
            if i + 1 == 7:
                root_fen = board.fen()
            if i + 1 >= min_depth + 7:
                x["root_fen"] = root_fen
                x["current_depth"] = i + 1 - 7
                boards.append(board.copy(stack=7))
                infos.append(x)
    return boards, infos


def make_gen_dict(dataset, wrapper, cache_hook, batch_size, min_depth):
    gen_dict = {}
    splits = ["train", "test"]

    def collate_fn(batch):
        return param_collate_fn(batch, min_depth)

    dataloaders = {
        split: DataLoader(
            dataset[split],
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        for split in splits
    }
    make_gen_list(gen_dict, dataloaders, wrapper, cache_hook)
    return {config: {split: merge_gens(gen_dict[config][split]) for split in splits} for config in gen_dict}


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
        filtered_dataset = dataset.filter(lambda s: s["depth"] >= args.min_depth)
        cache_hook = CacheHook(HookConfig(module_exp=r".*block\d+$"))
        cache_hook.register(wrapper)
        gen_dict = make_gen_dict(filtered_dataset, wrapper, cache_hook, args.batch_size, args.min_depth)
        for sub_config in gen_dict:
            dataset_dict = DatasetDict(
                {split: Dataset.from_generator(gen_dict[sub_config][split]) for split in ["train", "test"]}
            )
            full_config = f"{config_name}_{sub_config}"

            logger.info(f"Processed dataset: {dataset_dict}")

            if args.push_to_hub:
                logger.info("Pushing to hub...")
                dataset_dict.push_to_hub(
                    repo_id=args.dataset_name,
                    token=HF_TOKEN,
                    config_name=full_config,
                )


@dataclass
class Args:
    source_dataset: str = "Xmaster6y/lczero-planning-trajectories"
    source_config: str = "lc0-10-4238.onnx-policy"
    dataset_name: str = "Xmaster6y/lczero-planning-activations"
    push_to_hub: bool = False
    model_category: str = "small"
    use_all_models: bool = False
    min_depth: int = 10
    batch_size: int = 1000


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
    parser.add_argument("--min_depth", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
