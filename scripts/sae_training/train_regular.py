"""Train a SAE

Run with:
```bash
poetry run python -m scripts.sae_training.train_regular
```
"""

import argparse
from typing import Any, Dict

import torch
import wandb
from huggingface_hub import HfApi
from loguru import logger

from scripts.constants import HF_TOKEN, WANDB_API_KEY
from .run_regular import RegularRunConfig, make_regular_run


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hf_api = HfApi(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)  # type: ignore


def main(args):
    logger.info(f"Running on {DEVICE}")
    best_params: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "log_steps": args.log_steps,
        "val_steps": args.val_steps,
        "n_epochs": args.n_epochs,
        "beta1": args.beta1,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "lr": args.lr,
        "warmup_steps": 1_000,
        "cooldown_steps": 1_000,
        "dict_size_scale": args.dict_size_scale,
        "sparsity_loss_type": "d-l1",
        "sparsity_penalty_warmup_steps": 1_000,
    }
    run_params = best_params.copy()
    run_params.update({"source_dataset": args.source_dataset, "source_config": args.source_config})
    run_config = RegularRunConfig(
        **run_params,
    )
    save_folder = f"./assets/saes/{args.source_config}"
    with wandb.init(entity="yp-edu", project="contrastive-saes", config=run_params) as wandb_run:
        try:
            make_regular_run(run_config, wandb_run=wandb_run, save_folder=save_folder, streaming=args.streaming)
        except ValueError:
            run_config.beta1 = 0.9
            make_regular_run(run_config, wandb_run=wandb_run, save_folder=save_folder, streaming=args.streaming)
    if args.push_to_hub:
        hf_api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=args.source_config,
            folder_path=save_folder,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-one")
    parser.add_argument("--source_dataset", type=str, default="lczero-planning/activations")
    parser.add_argument(
        "--source_config",
        type=str,
        default="debug",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="lczero-planning/saes",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--dict_size_scale", type=int, default=8)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
