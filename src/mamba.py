# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None



# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)



class MultiLayerMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int = 2,
        rms_norm: bool = False,
        bi: bool = False,
    ):
        # init
        super().__init__()

        def create_block():
            norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5)
            if bi:
                MambaV2 = partial(Mamba, bimamba_type="v2")
                block = Block(d_model, MambaV2, norm_cls=norm_cls)
            else:
                block = Block(d_model, Mamba, norm_cls=norm_cls)
            return block

        self.layers = nn.ModuleList([create_block() for _ in range(n_layer)])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=1e-5)
        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x, inference_params=None):
        # x: [batch, length, dim]
        residual = None
        for layer in self.layers:
            x, residual = layer(
                x, residual, inference_params=inference_params
            )
        return x