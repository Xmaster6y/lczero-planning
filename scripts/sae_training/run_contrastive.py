"""Config for training a model."""

import os
from typing import Optional

import einops
import torch
from pydantic import BaseModel
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader

from scripts.sae_utils.training import trainSAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveRunConfig(BaseModel):
    # Unsweepable
    source_dataset: str
    source_config: str
    from_checkpoint: Optional[str] = None
    from_name: Optional[str] = None
    dict_size: Optional[int] = None
    batch_size: int
    log_steps: int
    val_steps: int
    save_steps: Optional[int] = None
    save_dir: Optional[str] = None
    # Sweepable
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    n_epochs: int
    warmup_steps: int
    cooldown_steps: int
    # SAE specific
    dict_size_scale: int
    ghost_threshold: Optional[int] = None
    resample_steps: Optional[int] = None
    sparsity_penalty_target: float
    sparsity_loss_type: str
    sparsity_penalty_warmup_steps: int
    contrastive_penalty: float
    contrastive_loss_type: str
    pre_bias: Optional[bool] = None
    init_normalise_dict: Optional[str] = None


def make_contrastive_run(
    run_config: ContrastiveRunConfig,
    wandb_run,
    save_folder: str,
    streaming: bool = False,
):
    logger.info(f"Running on {DEVICE}")

    init_ds = load_dataset(run_config.source_dataset, run_config.source_config, split="train", streaming=streaming)
    torch_ds = init_ds.select_columns(["root_act", "opt_act", "sub_act"]).with_format("torch")

    def map_fn(s_batched):
        b, c, h, w = s_batched["root_act"].shape
        new_s_batched = {}
        for act_type in ["root_act", "opt_act", "sub_act"]:
            new_s_batched[act_type] = einops.rearrange(s_batched[act_type], "b c h w -> (b h w) c")
        new_s_batched["pixel_index"] = einops.repeat(torch.arange(h * w), "(hw) -> (b hw) ", b=b)
        return new_s_batched

    dataset = torch_ds.map(map_fn, batched=True)
    if streaming:
        dataset = dataset.shuffle(seed=42)
        train_ds = dataset.filter(lambda x, i: i % 10 != 0, with_indices=True)
        val_ds = dataset.filter(lambda x, i: i % 10 == 0, with_indices=True)
    else:
        splitted_ds = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds = splitted_ds["train"]
        val_ds = splitted_ds["test"]

    train_dataloader = DataLoader(
        train_ds,
        batch_size=run_config.batch_size,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=run_config.batch_size,
    )
    if streaming:
        act_dim = next(iter(train_ds))["root_act"].shape[0] * 2
    else:
        act_dim = train_ds[0]["root_act"].shape[0] * 2
    dict_size = run_config.dict_size or run_config.dict_size_scale * act_dim
    sae = trainSAE(
        train_dataloader,
        act_dim,
        dict_size,
        lr=run_config.lr,
        beta1=run_config.beta1,
        beta2=run_config.beta2,
        weight_decay=run_config.weight_decay,
        n_epochs=run_config.n_epochs,
        warmup_steps=run_config.warmup_steps,
        cooldown_steps=run_config.cooldown_steps,
        val_dataloader=val_dataloader,
        sparsity_penalty_target=run_config.sparsity_penalty_target,
        sparsity_loss_type=run_config.sparsity_loss_type,
        sparsity_penalty_warmup_steps=run_config.sparsity_penalty_warmup_steps,
        contrastive_penalty=run_config.contrastive_penalty,
        contrastive_loss_type=run_config.contrastive_loss_type,
        pre_bias=run_config.pre_bias,
        init_normalise_dict=run_config.init_normalise_dict,
        resample_steps=run_config.resample_steps,
        ghost_threshold=run_config.ghost_threshold,
        save_steps=run_config.save_steps,
        val_steps=run_config.val_steps,
        save_dir=run_config.save_dir,
        log_steps=run_config.log_steps,
        device=DEVICE,
        from_checkpoint=run_config.from_checkpoint,
        wandb_run=wandb_run,
        do_print=True,
    )

    os.makedirs(save_folder, exist_ok=True)
    if run_config.from_checkpoint:
        model_path = f"{save_folder}/from_{run_config.from_name}.pt"
    else:
        prefix = "contrastive" if run_config.contrastive_penalty > 0 else "contrastive_no_penalty"
        model_path = f"{save_folder}/{prefix}_model.pt"
    torch.save(
        sae.state_dict(),
        model_path,
    )
    logger.info(f"Model saved to {model_path}")
