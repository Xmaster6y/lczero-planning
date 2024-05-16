"""Make a dataset of features from a source dataset.

Run with:
```bash
poetry run python -m scripts.datasets.make_features_dataset
```
"""

import argparse
from dataclasses import dataclass

import torch
from huggingface_hub import HfApi
from loguru import logger
from datasets import load_dataset
import einops

from scripts.constants import HF_TOKEN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hf_api = HfApi(token=HF_TOKEN)


def main(args):
    logger.info(f"Running on {DEVICE}")
    hf_api.snapshot_download(repo_id=args.repo_id, repo_type="model", local_dir="./assets/saes")
    ae = torch.load(
        f"./assets/saes/{args.source_config}/model.pt",
        map_location=DEVICE,
    )

    init_ds = load_dataset(args.source_dataset, args.source_config, split="test")
    torch_ds = init_ds.with_format("torch")

    def map_fn(s_batched):
        b, c, h, w = s_batched["root_act"].shape
        new_s_batched = {}
        for act_type in ["root_act", "opt_act", "sub_act"]:
            new_s_batched[act_type] = einops.rearrange(s_batched.pop(act_type), "b c h w -> (b h w) c")
        new_s_batched["pixel_index"] = einops.repeat(torch.arange(h * w), "hw -> (b hw) ", b=b)
        for k, v in s_batched.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 2:
                    new_s_batched[k] = einops.repeat(v, "b r -> (b hw) r", hw=h * w)
                else:
                    new_s_batched[k] = einops.repeat(v, "b -> (b hw) ", hw=h * w)
            else:
                new_s_batched[k] = v * (h * w)
        return new_s_batched

    dataset = torch_ds.map(map_fn, batched=True)

    def compute_features_fn(batch, ae):
        root_activations = batch["root_act"]
        opt_activations = batch["opt_act"]
        sub_activations = batch["sub_act"]
        activations = torch.cat([root_activations, opt_activations], dim=1)
        activations = activations.to(DEVICE)
        f_pre = ae.encode(activations)
        opt_f = ae.relu(f_pre)
        activations = torch.cat([root_activations, sub_activations], dim=1)
        activations = activations.to(DEVICE)
        f_pre = ae.encode(activations)
        sub_f = ae.relu(f_pre)
        return {"opt_features": opt_f, "sub_features": sub_f}

    features_dataset = dataset.map(
        compute_features_fn,
        batched=True,
        remove_columns=["root_act", "opt_act", "sub_act"],
        fn_kwargs={"ae": ae},
    )

    if args.push_to_hub:
        features_dataset.push_to_hub(
            args.source_dataset.replace("activations", "features"), args.source_config, split="test"
        )


@dataclass
class Args:
    source_dataset: str = "Xmaster6y/lczero-planning-activations"
    source_config: str = "debug"
    repo_id: str = "Xmaster6y/lczero-planning-saes"
    push_to_hub: bool = False
    batch_size: int = 1000


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
