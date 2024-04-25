# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# temp_config = {
#     "bi": False,
#     "attention": True,
#     "conv": True,
#     "d_conv": 1,
#     "conv_group": 1 
# }
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_devide_out=False,
        init_layer_scale=None,
        bi = False,
        attention = True,
        conv = True,
        conv_group = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.bi = bi
        self.if_devide_out = if_devide_out
        self.attention = attention
        self.conv = conv
        self.conv_group = conv_group
        self.init_layer_scale = init_layer_scale
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.act = nn.SiLU()
        out_in_channels = self.d_inner*2 if bi else self.d_inner
        self.out_proj = nn.Linear(out_in_channels, self.d_model, bias=bias, **factory_kwargs)
        # forward and backward
        type_lst = ["fwd"] if not self.bi else ["fwd", "bwd"]
        for typ in type_lst:
            # Convolution, activated when kernel size <= 1,
            setattr(self, f"conv1d_{typ}", nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner if self.d_conv > 1 else 1,
                padding=d_conv - 1,
                bias=conv_bias,
                **factory_kwargs
                ))
            # S4D real initialization
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner
            ).contiguous()

            A_log = torch.log(A)  # Keep A_log in fp32
            A_log_param = nn.Parameter(A_log)
            setattr(self, f"A_log_{typ}", A_log_param)
            
            # x ---> B, C, Delta
            setattr(self, f"x_proj_{typ}", 
                    nn.Linear(
                        self.d_inner, 
                        self.dt_rank + self.d_state * 2, 
                        bias=False, 
                        **factory_kwargs))
            # Delta
            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            setattr(self, f"dt_proj_{typ}", dt_proj)
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.data.copy_(inv_dt)
            # D "skip" parameter
            D_param = nn.Parameter(torch.ones(self.d_inner, device=device))
            setattr(self, f"D_{typ}", D_param)
            
            # Set no weight decay
            A_log_param._no_weight_decay = True
            D_param._no_weight_decay = True

            # Mark dt_proj.bias as no reinit
            dt_proj.bias._no_reinit = True

    def conv_ssm(self, x, z, typ):
        seqlen = x.shape[-1]
        # Compute convolution when kernel size <= 1
        if self.conv:
                x = self.act(getattr(self, f"conv1d_{typ}")(x))
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        # A
        A = -torch.exp(getattr(self, f"A_log_{typ}").float())
        # x_proj, dt_proj, A
        x_dbl = getattr(self, f"x_proj_{typ}")(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = getattr(self, f"dt_proj_{typ}").weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            getattr(self, f"D_{typ}").float(),
            z=z,
            delta_bias=getattr(self, f"dt_proj_{typ}").bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = rearrange(y, "b d l -> b l d")
        return y

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        # xz
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # if (self.d_conv <= 1) or (not self.conv):
        # Bi-directional Mamba (Manual)
        # Single-directional Mamba (Manual)
        x, z = xz.chunk(2, dim=1)
        z = None if not self.attention else z
        if self.bi:
            # x [b d l]
            # y [b l d]
            y_fwd = self.conv_ssm(x, z, "fwd")
            y_bwd = self.conv_ssm(x.flip([-1]), z.flip([-1]) if self.attention else None, "bwd")
            # concat to b l 2d
            y = torch.cat((y_fwd, y_bwd.flip([-1])), dim=2)
            # y --> out [b l d]
            out = self.out_proj(y)
        else:
            y = self.conv_ssm(x, z, "fwd")
            out = self.out_proj(y)
        # else:
        #     # bi-directional Mamba (Fast path)
        #     if self.bi:
        #         # x, z = xz.chunk(2, dim=1)
        #         out_list = list()
        #         for typ in ["fwd", "bwd"]:
        #             xz_input = xz if typ == "fwd" else xz.flip([-1])
        #             A = -torch.exp(self.param[f"A_log_{typ}"].float())  # (d_inner, d_state)
                    
        #             mamba_out = mamba_inner_fn_no_out_proj(
        #                 xz_input,
        #                 self.param[f"conv1d_{typ}"].weight,
        #                 self.param[f"conv1d_{typ}"].bias,
        #                 self.param[f"x_proj_{typ}"].weight,
        #                 self.param[f"dt_proj_{typ}"].weight,
        #                 A,
        #                 None,  # input-dependent B
        #                 None,  # input-dependent C
        #                 self.param[f"D_{typ}"].float(),
        #                 delta_bias=self.param[f"dt_proj_{typ}"].bias.float(),
        #                 delta_softplus=True,
        #                 )
        #             out_list.append(mamba_out)

        #         out, out_b = out_list   
        #         # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        #         if not self.if_devide_out:
        #             out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        #         else:
        #             out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)
        #     # single-directional Mamba (Fast path)
        #     else:
        #         A = -torch.exp(self.param["A_log_fwd"].float())
        #         out = mamba_inner_fn(
        #             xz,
        #             self.param["conv1d_fwd"].weight,
        #             self.param["conv1d_fwd"].bias,
        #             self.param["x_proj_fwd"].weight,
        #             self.param["dt_proj_fwd"].weight,
        #             self.out_proj.weight,
        #             self.out_proj.bias,
        #             A,
        #             None,  # input-dependent B
        #             None,  # input-dependent C
        #             self.param["D_fwd"].float(),
        #             delta_bias=self.param["dt_proj_fwd"].bias.float(),
        #             delta_softplus=True,
        #         )
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state



class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
