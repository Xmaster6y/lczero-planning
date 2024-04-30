"""Script to generate the base datasets.

Run with:
```bash
poetry run python -m scripts.datasets.convert_tcec_dataset
```
"""

import argparse

from datasets import load_dataset
from loguru import logger
from lczerolens.game import preprocess

from scripts.constants import HF_TOKEN


def map_fn(
    batch,
    n_history,
    skip_book_exit,
    skip_first_n,
):
    output_batch = {"fen": [], "gameid": [], "moves": []}
    for gameid, moves in zip(batch["gameid"], batch["moves"]):
        boards = preprocess.game_to_boards(
            preprocess.dict_to_game({"gameid": gameid, "moves": moves}),
            n_history,
            skip_book_exit,
            skip_first_n,
            output_dict=True,
        )
        for board in boards:
            for k in output_batch:
                output_batch[k].append(board[k])
    return output_batch


def main(args: argparse.Namespace):
    logger.info(f"Loading `{args.source_dataset}`...")
    dataset = load_dataset(args.source_dataset, split="train")

    logger.info(f"Processing dataset: {dataset}")
    processed_ds = dataset.map(
        map_fn,
        fn_kwargs={
            "n_history": args.n_history,
            "skip_book_exit": args.skip_book_exit,
            "skip_first_n": args.skip_first_n,
        },
        batched=True,
    )
    logger.info(f"Processed dataset: {processed_ds}")
    splitted_ds = processed_ds.train_test_split(
        test_size=args.test_size,
        seed=args.seed,
    )
    logger.info(f"Splitted dataset: {splitted_ds}")

    if args.push_to_hub:
        logger.info("Pushing to hub...")
        splitted_ds.push_to_hub(
            repo_id=args.dataset_name,
            token=HF_TOKEN,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("convert-tcec-dataset")
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="Xmaster6y/lczero-planning-tcec",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Xmaster6y/lczero-planning-boards",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-history", type=int, default=7)
    parser.add_argument("--skip-book-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-first-n", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
