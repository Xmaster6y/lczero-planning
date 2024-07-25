<img src="./assets/figures/lczero-planning_thumbnail.png" alt="dynamical concepts" width="200"/>

# Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents

[**Project page**](https://yp-edu.github.io/publications/lczero-planning) | [**HF Space**](https://huggingface.co/spaces/lczero-planning/demo) | [**Paper**](http://arxiv.org/abs/2406.04028)

Scripts for interpreting planning in LeelaChessZero networks.

:red_circle: __*Not a stable codebase*__

## Install & Run

This work relies on poetry to manage the dependencies. To install run (with additional `demo` group for running the demo):

```
poetry install
```

Then to run a particular script use:

```
poetry run python -m scripts.sae_training.train_contrastive
```

To run the demo you can use the following `make` shortcut:

```
make demo
```

## Tooling

See the [lczerolens](https://github.com/Xmaster6y/lczerolens) library (still under development) for more agnostic tooling to interpret the Leela Networks.

## Contribute

Feel free to open a discussion, an issue or a PR for any question or feedback.

## Cite

If you find this work useful please consider citing the associated paper:

```
@misc{poupart2024contrastivesparseautoencodersinterpreting,
      title={Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents},
      author={Yoann Poupart},
      year={2024},
      eprint={2406.04028},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.04028},
}
```
