"""Constants for the scripts."""

import os

# Models
BIG_MODELS = [
    "lc0-19-4508.onnx",
    "lc0-19-4000.onnx",
    "lc0-19-3056.onnx",
    "lc0-19-1876.onnx",
]
MEDIUM_MODELS = [
    "lc0-15-3350.onnx",
    "lc0-15-3023.onnx",
    "lc0-15-1815.onnx",
]
SMALL_MODELS = [
    "lc0-10-4238.onnx",
    "lc0-10-4012.onnx",
    "lc0-10-3051.onnx",
    "lc0-10-1893.onnx",
]
SMALL_REF_MODEL = "lc0-10-4238.onnx"
REF_MODEL = "lc0-19-4508.onnx"

# Secrets
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
