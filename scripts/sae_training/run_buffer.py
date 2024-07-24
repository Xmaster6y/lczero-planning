"""Config for training a model."""

import os
from typing import Optional

import chess
import copy
import einops
import torch
from pydantic import BaseModel
from loguru import logger
from datasets import load_dataset

from lczerolens import LczeroModel
from lczerolens.lenses import ActivationLens, ActivationBuffer

from scripts.sae_utils.buffer_training import trainSAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BufferRunConfig(BaseModel):
    # Unsweepable
    source_dataset: str
    source_config: str
    from_checkpoint: Optional[str] = None
    from_name: Optional[str] = None
    dict_size: Optional[int] = None
    batch_size: int
    module_name: str
    run_name: str
    act_dim: int
    n_batches_in_buffer: int
    compute_batch_size: int
    min_depth: int
    log_steps: int
    val_steps: int
    save_steps: Optional[int] = None
    save_dir: Optional[str] = None
    # Sweepable
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    max_train_batches: int
    max_val_batches: int
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
    use_contrastive: bool
    contrastive_penalty: float
    contrastive_loss_type: str
    pre_bias: Optional[bool] = None
    init_normalise_dict: Optional[str] = None


def make_buffer_run(
    run_config: BufferRunConfig,
    wandb_run,
    save_folder: str,
):
    logger.info(f"Running on {DEVICE}")
    model_name, _ = run_config.source_config.split(".onnx")
    model = LczeroModel.from_path(f"assets/{model_name}.onnx").to(DEVICE)

    dataset = load_dataset(run_config.source_dataset, run_config.source_config)

    lens = ActivationLens(run_config.module_name)

    def collate_fn(batch):
        return _collate_fn(
            batch,
            run_config.min_depth,
        )

    def compute_fn(batch, model):
        return _compute_fn(
            batch,
            model,
            run_config.use_contrastive,
            lens,
        )

    train_buffer = ActivationBuffer(
        model,
        dataset["train"],
        compute_fn,
        run_config.n_batches_in_buffer,
        run_config.compute_batch_size,
        run_config.batch_size,
        dataloader_kwargs={"collate_fn": collate_fn, "shuffle": True},
        logger=logger,
    )

    val_buffer = ActivationBuffer(
        model,
        dataset["test"],
        compute_fn,
        run_config.n_batches_in_buffer,
        run_config.compute_batch_size,
        run_config.batch_size,
        dataloader_kwargs={"collate_fn": collate_fn, "shuffle": True},
        logger=logger,
    )

    dict_size = run_config.dict_size or run_config.dict_size_scale * run_config.act_dim
    sae = trainSAE(
        train_buffer,
        run_config.act_dim,
        dict_size,
        lr=run_config.lr,
        beta1=run_config.beta1,
        beta2=run_config.beta2,
        weight_decay=run_config.weight_decay,
        n_epochs=run_config.n_epochs,
        max_train_batches=run_config.max_train_batches,
        max_val_batches=run_config.max_val_batches,
        warmup_steps=run_config.warmup_steps,
        cooldown_steps=run_config.cooldown_steps,
        val_buffer=val_buffer,
        sparsity_penalty_target=run_config.sparsity_penalty_target,
        sparsity_loss_type=run_config.sparsity_loss_type,
        sparsity_penalty_warmup_steps=run_config.sparsity_penalty_warmup_steps,
        use_contrastive=run_config.use_contrastive,
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
        if run_config.use_contrastive:
            prefix = "contrastive" if run_config.contrastive_penalty > 0 else "contrastive_no_penalty"
        else:
            prefix = "regular"
        model_path = f"{save_folder}/{prefix}{run_config.run_name}_model.pt"
    torch.save(
        sae.state_dict(),
        model_path,
    )
    logger.info(f"Model saved to {model_path}")


def _collate_fn(batch, min_depth):
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


def channel_transform(*, opt_acts, root_acts=None, sub_acts=None, contrastive=False):
    new_opt_acts = einops.rearrange(opt_acts, "b c h w -> (b h w) c")
    if contrastive:
        new_sub_acts = einops.rearrange(sub_acts, "b c h w -> (b h w) c")
        new_root_acts = einops.rearrange(root_acts, "b c h w -> (b h w) c")
        d_acts = torch.cat([new_opt_acts, new_sub_acts], dim=0)
        acts = torch.cat([new_root_acts.repeat(2, 1), d_acts], dim=1)
        return acts
    return new_opt_acts


def format_activations(acts, infos, contrastive):
    if contrastive:
        prev_root = None
        prev_opt = None
        root_acts = []
        opt_acts = []
        sub_acts = []
        for act, info in zip(acts, infos):
            is_root = info.pop("is_root")
            if is_root:
                prev_root = (act, info)
                prev_opt = None
                continue
            if "opt_fen" in info:
                prev_opt = (act, info)
                continue
            root_act, root_info = prev_root
            opt_act, opt_info = prev_opt
            root_acts.append(root_act)
            opt_acts.append(opt_act)
            sub_acts.append(act)
        return channel_transform(
            root_acts=torch.stack(root_acts),
            opt_acts=torch.stack(opt_acts),
            sub_acts=torch.stack(sub_acts),
            contrastive=contrastive,
        )

    return channel_transform(opt_acts=acts, contrastive=contrastive)


def _compute_fn(batch, model, contrastive, lens):
    boards, infos = batch
    storage = lens.analyse(*boards, model=model)[0]
    if len(storage.keys()) != 1:
        raise NotImplementedError
    acts = next(iter(storage.values()))
    return format_activations(acts, infos, contrastive)
