"""
Defines the dictionary classes
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class SparseAutoEncoder(nn.Module):
    """
    A 2-layer sparse autoencoder.
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        pre_bias=False,
        init_normalise_dict=None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.pre_bias = pre_bias
        self.init_normalise_dict = init_normalise_dict

        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))
        self.relu = nn.ReLU()

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.dict_size,
                    self.activation_dim,
                )
            )
        )
        if init_normalise_dict == "l2":
            self.normalize_dict_(less_than_1=False)
            self.W_dec *= 0.1
        elif init_normalise_dict == "less_than_1":
            self.normalize_dict_(less_than_1=True)

        self.W_enc = nn.Parameter(self.W_dec.t())
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.activation_dim,
            )
        )

    @torch.no_grad()
    def normalize_dict_(
        self,
        less_than_1=False,
    ):
        norm = self.W_dec.norm(dim=1)
        positive_mask = norm != 0
        if less_than_1:
            greater_than_1_mask = (norm > 1) & (positive_mask)
            self.W_dec[greater_than_1_mask] /= norm[greater_than_1_mask].unsqueeze(1)
        else:
            self.W_dec[positive_mask] /= norm[positive_mask].unsqueeze(1)

    def encode(self, x):
        return x @ self.W_enc + self.b_enc

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well
            as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode"
            where features are masked
        """
        if self.pre_bias:
            x = x - self.b_dec
        f_pre = self.encode(x)
        out = TensorDict({}, batch_size=x.shape[0])
        if ghost_mask is not None:
            f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
            x_ghost = f_ghost @ self.W_dec
            out["x_ghost"] = x_ghost
        f = self.relu(f_pre)
        if output_features:
            out["features"] = f
        x_hat = self.decode(f)
        out["x_hat"] = x_hat
        return out
