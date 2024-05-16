"""Train a SAE

Run with:
```bash
poetry run python -m scripts.sae_training.train_contrastive
```
"""

import argparse
from typing import Any, Dict

import torch
import wandb
from huggingface_hub import HfApi
from loguru import logger

from scripts.constants import HF_TOKEN, WANDB_API_KEY
from .run_contrastive import ContrastiveRunConfig, make_contrastive_run


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hf_api = HfApi(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)  # type: ignore


def main(args):
    logger.info(f"Running on {DEVICE}")
    best_params: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "log_steps": args.log_steps,
        "val_steps": args.val_steps,
        "n_epochs": 50,
        "beta1": 0.0,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "lr": args.lr,
        "warmup_steps": 1_000,
        "cooldown_steps": 1_000,
        "dict_size_scale": args.dict_size_scale,
        "sparsity_penalty_target": 0.1,
        "sparsity_loss_type": "sq-l1",
        "sparsity_penalty_warmup_steps": 1_000,
        "contrastive_penalty": 1.0,
        "contrastive_loss_type": "diff-prod",
    }
    run_params = best_params.copy()
    run_params.update({"source_dataset": args.source_dataset, "source_config": args.source_config})
    run_config = ContrastiveRunConfig(
        **run_params,
    )
    save_folder = f"./assets/models/{args.source_config}"
    with wandb.init(entity="yp-edu", project="contrastive-saes", config=run_params) as wandb_run:
        try:
            make_contrastive_run(run_config, wandb_run=wandb_run, save_folder=save_folder)
        except ValueError:
            run_config.beta1 = 0.9
            make_contrastive_run(run_config, wandb_run=wandb_run, save_folder=save_folder)
    if args.push_to_hub:
        hf_api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=args.source_config,
            folder_path=save_folder,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-one")
    parser.add_argument("--source_dataset", type=str, default="Xmaster6y/lczero-planning-activations")
    parser.add_argument(
        "--source_config",
        type=str,
        default="debug",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Xmaster6y/lczero-planning-saes",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dict_size_scale", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
