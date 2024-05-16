"""
Training dictionaries
"""

import os

from loguru import logger
import torch as t
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import explained_variance_score, r2_score

from .sae import SparseAutoEncoder

EPS = 1e-8
t.autograd.set_detect_anomaly(True)


def entropy(p, eps=1e-8):
    p_sum = p.sum(dim=-1, keepdim=True)
    # epsilons for numerical stability
    p_normed = p / (p_sum + eps)
    p_log = t.log(p_normed + eps)
    ent = -(p_normed * p_log)

    # Zero out the entropy where p_sum is zero
    ent = t.where(p_sum > 0, ent, t.zeros_like(ent))

    return ent.sum(dim=-1).mean()


def ghost_loss_fn(out, activations):
    x_ghost = out["x_ghost"]
    x_hat = out["x_hat"]
    residual = activations - x_hat
    x_ghost = (
        x_ghost
        * residual.norm(dim=-1, keepdim=True).detach()
        / (2 * x_ghost.norm(dim=-1, keepdim=True).detach() + EPS)
    )
    return t.linalg.norm(residual.detach() - x_ghost, dim=-1).mean()


def sae_loss(
    acts,
    ae,
    sparsity_penalty,
    sparsity_loss_type=None,
    contrastive_penalty=0.1,
    contrastive_loss_type="diff-prod",
    num_samples_since_activated=None,
    ghost_threshold=None,
    explained_variance=False,
    r2=False,
):
    """
    Compute the loss of an autoencoder on some activations
    If num_samples_since_activated is not None, update it in place
    If ghost_threshold is not None, use it to do ghost grads
    """
    if isinstance(acts, dict):
        root_activations = acts["root_act"]
        opt_activations = acts["opt_act"]
        sub_activations = acts["sub_act"]
        d_activations = t.cat([opt_activations, sub_activations], dim=0)
        activations = t.cat([root_activations.repeat(2, 1), d_activations], dim=1)
        use_contrastive_loss = True
    else:
        use_contrastive_loss = False
        activations = acts

    if ghost_threshold is not None:
        if num_samples_since_activated is None:
            raise ValueError("num_samples_since_activated must be provided for ghost grads")
        ghost_mask = num_samples_since_activated > ghost_threshold
        if ghost_mask.sum() == 0:  # if there are dead neurons
            ghost_mask = None
    else:
        ghost_mask = None

    out = ae(activations, output_features=True, ghost_mask=ghost_mask)

    f = out["features"]
    x_hat = out["x_hat"]
    mse_loss = t.linalg.norm(activations - x_hat, dim=-1).mean()

    deads = (f == 0).all(dim=0)
    if num_samples_since_activated is not None:  # update the number of samples since each neuron was last activated
        num_samples_since_activated.copy_(t.where(deads, num_samples_since_activated + 1, 0))
        activated_loss = num_samples_since_activated.float().mean()
    else:
        activated_loss = 0

    l0_loss = (f > 0).sum(dim=-1).float().mean()
    dead_loss = deads.float().mean()
    if ghost_mask is not None:
        ghost_loss = ghost_loss_fn(out, activations)
    else:
        ghost_loss = None

    if sparsity_loss_type == "entropy":
        sparsity_loss = entropy(f)
    elif sparsity_loss_type == "d-l1":
        sparsity_loss = (f.abs() * ae.W_dec.norm(dim=-1)).sum(dim=-1).mean()
    else:
        sparsity_loss = f.norm(p=1, dim=-1).mean()

    out_losses = {
        "mse_loss": mse_loss,
        "sparsity_loss": sparsity_loss,
        "l0_loss": l0_loss,
        "dead_loss": dead_loss,
        "activated_loss": activated_loss,
    }
    classical_loss = mse_loss + sparsity_penalty * sparsity_loss
    if use_contrastive_loss:
        f_opt, f_sub = f.chunk(2, dim=0)
        c_f_opt, d_f_opt = f_opt.chunk(2, dim=1)
        c_f_sub, d_f_sub = f_sub.chunk(2, dim=1)
        c_type, d_type = contrastive_loss_type.split("-")

        if c_type == "diff":
            c_diff_loss = t.norm(c_f_opt - c_f_sub, p=1, dim=-1).mean()
        else:
            raise NotImplementedError(f"Contrastive loss type {contrastive_loss_type} not implemented")

        if d_type == "prod":
            d_prod_loss = t.norm(d_f_opt * d_f_sub, p=1, dim=-1).mean()
        else:
            raise NotImplementedError(f"Contrastive loss type {contrastive_loss_type} not implemented")

        contrastive_loss = c_diff_loss + d_prod_loss
        out_losses["contrastive_loss"] = contrastive_loss
        classical_loss += contrastive_penalty * contrastive_loss
    if ghost_loss is None:
        out_losses["total_loss"] = classical_loss
    else:
        out_losses["ghost_loss"] = ghost_loss
        out_losses["total_loss"] = classical_loss + ghost_loss * (mse_loss.detach() / (ghost_loss.detach() + EPS))
    if explained_variance:
        out_losses["explained_variance"] = explained_variance_score(activations.detach().cpu(), x_hat.detach().cpu())
    if r2:
        out_losses["r2_score"] = r2_score(activations.detach().cpu(), x_hat.detach().cpu())
    return out_losses


@t.no_grad
def resample_neurons(deads, activations, ae, optimizer):
    """
    resample dead neurons according to the following scheme:
    Reinitialize the decoder vector for each dead neuron to be an activation
    vector v from the dataset with probability proportional to ae's loss on v.
    Reinitialize all dead encoder vectors to be the mean alive encoder.
    Reset the bias vectors for dead neurons to 0.
    Reset the Adam parameters for the dead neurons to their default values.
    """
    if deads.sum() == 0:
        return
    if isinstance(activations, tuple):
        in_acts, out_acts = activations
    else:
        in_acts = out_acts = activations
    in_acts = in_acts.reshape(-1, in_acts.shape[-1])
    out_acts = out_acts.reshape(-1, out_acts.shape[-1])

    # compute the loss for each activation vector
    losses = (out_acts - ae(in_acts)["x_hat"]).norm(dim=-1)

    # resample decoder vectors for dead neurons
    indices = t.multinomial(losses, num_samples=deads.sum(), replacement=True)

    ae.W_dec[deads] = out_acts[indices]
    ae.normalize_dict_()

    # resample encoder vectors for dead neurons
    ae.W_dec[deads] = ae.W_dec[~deads].mean(dim=0, keepdim=True) * 0.2

    # reset Adam parameters for dead neurons
    state_dict = optimizer.state_dict()["state"]
    # # encoder weight
    state_dict[1]["exp_avg"][deads] = 0.0
    state_dict[1]["exp_avg_sq"][deads] = 0.0


def trainSAE(
    train_dataloader,
    activation_dim,
    dictionary_size,
    *,
    lr=1e-5,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0,
    n_epochs=1,
    warmup_steps=1000,
    cooldown_steps=1000,
    val_dataloader=None,
    sparsity_penalty_target=0.01,
    sparsity_loss_type="sq-l1",
    sparsity_penalty_warmup_steps=1000,
    contrastive_penalty=0.1,
    contrastive_loss_type="diff-prod",
    pre_bias=False,
    init_normalise_dict=None,
    resample_steps=None,
    ghost_threshold=None,
    save_steps=None,
    val_steps=None,
    save_dir=None,
    log_steps=1000,
    device="cpu",
    from_checkpoint=None,
    wandb_run=None,
    do_print=True,
):
    """
    Train and return a sparse autoencoder
    """
    ae = SparseAutoEncoder(
        activation_dim, dictionary_size, pre_bias=pre_bias, init_normalise_dict=init_normalise_dict
    ).to(device)
    if from_checkpoint is not None:
        loaded = t.load(from_checkpoint)
        if isinstance(loaded, t.nn.Module):
            ae.load_state_dict(loaded.state_dict())
        else:
            ae.load_state_dict(loaded)

    num_samples_since_activated = t.zeros(dictionary_size, dtype=int).to(
        device
    )  # how many samples since each neuron was last activated?

    # set up optimizer and scheduler
    optimizer = t.optim.AdamW(
        ae.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    total_steps = n_epochs * len(train_dataloader)

    def lr_fn(step):
        # cooldown
        cooldown_ratio = min(1.0, (total_steps - step) / cooldown_steps)
        # warmup
        if resample_steps is not None:
            ini_step = step % resample_steps
        else:
            ini_step = step
        return min(ini_step / warmup_steps, 1.0) * cooldown_ratio

    def sparsity_penalty_fn(step):
        # warmup
        return min(step / sparsity_penalty_warmup_steps, 1.0) * sparsity_penalty_target

    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    step = 0
    for _ in range(n_epochs):
        for acts in train_dataloader:
            step += 1

            if isinstance(acts, t.Tensor):  # typical casse
                acts = acts.to(device)
            elif isinstance(acts, dict):
                acts = {k: v.to(device) for k, v in acts.items()}

            optimizer.zero_grad()
            # updates num_samples_since_activated in place
            sparsity_penalty = sparsity_penalty_fn(step)
            losses = sae_loss(
                acts,
                ae,
                sparsity_penalty,
                sparsity_loss_type=sparsity_loss_type,
                contrastive_penalty=contrastive_penalty,
                contrastive_loss_type=contrastive_loss_type,
                num_samples_since_activated=num_samples_since_activated,
                ghost_threshold=ghost_threshold,
            )
            losses["total_loss"].backward()
            clip_grad_norm_(ae.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            ae.normalize_dict_()

            # deal with resampling neurons
            if resample_steps is not None and step % resample_steps == 0:
                resample_neurons(
                    num_samples_since_activated > resample_steps / 2,
                    acts,
                    ae,
                    optimizer,
                )

            # logging
            if log_steps is not None and step % log_steps == 0:
                with t.no_grad():
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"train/{k}": v for k, v in losses.items()},
                            step=step,
                        )
                    if do_print:
                        logger.info(f"Train step {step}: {losses}")
            if save_steps is not None and save_dir is not None and step % save_steps == 0:
                if not os.path.exists(os.path.join(save_dir, "checkpoints")):
                    os.mkdir(os.path.join(save_dir, "checkpoints"))
                t.save(
                    ae.state_dict(),
                    os.path.join(save_dir, "checkpoints", f"ae_{step}.pt"),
                )
            if val_steps is not None and val_dataloader is not None:
                with t.no_grad():
                    if step % val_steps == 0:
                        val_losses = {}
                        for val_acts in val_dataloader:
                            if isinstance(val_acts, t.Tensor):
                                val_acts = val_acts.to(device)
                            elif isinstance(val_acts, dict):
                                val_acts = {k: v.to(device) for k, v in val_acts.items()}
                            losses = sae_loss(
                                val_acts,
                                ae,
                                sparsity_penalty,
                                sparsity_loss_type=sparsity_loss_type,
                                num_samples_since_activated=(num_samples_since_activated),
                                ghost_threshold=ghost_threshold,
                                explained_variance=True,
                                r2=True,
                            )
                            for k, _ in val_losses.items():
                                if k not in val_losses:
                                    val_losses[k] = 0
                                val_losses[k] += losses[k] / len(val_dataloader)

                        if wandb_run is not None:
                            wandb_run.log(
                                {f"val/{k}": v for k, v in val_losses.items()},
                                step=step,
                            )
                        if do_print:
                            logger.info(f"Val step {step}: {val_losses}")

    return ae
