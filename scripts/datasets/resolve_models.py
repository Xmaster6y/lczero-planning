"""Script to resolve models.

Run with:
```bash
poetry run python -m scripts.datasets.resolve_models
```
"""

from huggingface_hub import HfApi

from scripts.constants import HF_TOKEN

hf_api = HfApi(token=HF_TOKEN)

hf_api.snapshot_download(
    "lczero-planning/lczero-planning-models",
    repo_type="model",
    local_dir="assets/models",
)
