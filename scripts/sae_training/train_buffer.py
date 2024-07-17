"""Train a SAE

Run with:
```bash
poetry run python -m scripts.sae_training.train_buffer
```
"""

import argparse
from typing import Any, Dict

import torch
import wandb
from huggingface_hub import HfApi
from loguru import logger

from scripts.constants import HF_TOKEN, WANDB_API_KEY
from .run_buffer import BufferRunConfig, make_buffer_run


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hf_api = HfApi(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)  # type: ignore


def main(args):
    logger.info(f"Running on {DEVICE}")
    best_params: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "module_name": args.module_name,
        "act_dim": args.act_dim,
        "n_batches_in_buffer": args.n_batches_in_buffer,
        "compute_batch_size": args.compute_batch_size,
        "min_depth": args.min_depth,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
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
        "sparsity_penalty_target": args.sparsity_penalty_target,
        "sparsity_loss_type": "d-l1",
        "sparsity_penalty_warmup_steps": 1_000,
        "use_contrastive": args.use_contrastive,
        "contrastive_penalty": args.contrastive_penalty,
        "contrastive_loss_type": "diff-prod",
    }
    run_params = best_params.copy()
    run_params.update({"source_dataset": args.source_dataset, "source_config": args.source_config})
    run_config = BufferRunConfig(
        **run_params,
    )
    save_folder = f"./assets/saes/{args.source_config}"
    with wandb.init(entity="yp-edu", project="buffer-saes", config=run_params) as wandb_run:
        try:
            make_buffer_run(run_config, wandb_run=wandb_run, save_folder=save_folder)
        except ValueError:
            run_config.beta1 = 0.9
            make_buffer_run(run_config, wandb_run=wandb_run, save_folder=save_folder)
    if args.push_to_hub:
        hf_api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=args.source_config,
            folder_path=save_folder,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-one")
    parser.add_argument("--source_dataset", type=str, default="lczero-planning/trajectories")
    parser.add_argument(
        "--source_config",
        type=str,
        default="lc0-10-4238.onnx-policy",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="lczero-planning/saes",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--module_name", type=str, default="block9/conv2/relu")
    parser.add_argument("--act_dim", type=int, default=128)
    parser.add_argument("--n_batches_in_buffer", type=int, default=10)
    parser.add_argument("--compute_batch_size", type=int, default=100)
    parser.add_argument("--min_depth", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=1_000)
    parser.add_argument("--max_val_batches", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--use_contrastive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contrastive_penalty", type=float, default=0.001)
    parser.add_argument("--sparsity_penalty_target", type=float, default=0.001)
    parser.add_argument("--dict_size_scale", type=int, default=8)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
