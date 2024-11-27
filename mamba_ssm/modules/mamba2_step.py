# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


import os
import sys
fscil_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(fscil_directory,"../../quantstudy"))
from quantstudy.models.quant_tools import QuantLinear, act_quant_fn, weight_quant_fn


class Mamba2(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, conv_init=None, expand=2, headdim=64, ngroups=1, A_init_range=(1, 16),
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, dt_limit=(0.0, float("inf")), bias=False, conv_bias=True,
                 chunk_size=256, use_mem_eff_path=True, layer_idx=None, process_group=None, sequence_parallel=True, device=None,
                 dtype=None, quant_bits=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.dt_limit = dt_limit
        self.activation = "relu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.quant = (quant_bits is not None)
        self.quant_bits = quant_bits
        self.taylor_exp = True
        self.no_sftplus = True

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        self.in_proj = QuantLinear(self.d_model, d_in_proj, bias=bias, 
                                   act_bits=quant_bits['act'], w_bits=quant_bits['weights'])

        self.rec_dt_proj = nn.Linear(self.d_inner * self.d_state, self.nheads, bias=bias, **factory_kwargs)

        fan_in = self.d_inner * self.d_state
        bound = 1 / (10 * math.sqrt(fan_in))
        nn.init.uniform_(self.rec_dt_proj.weight, -bound, bound)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding='same',
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.ReLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm, device=device))
        self.D._no_weight_decay = True

        self.out_proj = QuantLinear(self.d_inner, self.d_model, bias=bias, 
                                    act_bits=quant_bits['act'], w_bits=quant_bits['weights'])

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        if self.quant:
            act_levels = 2**(self.quant_bits['act']-1)-1
            xBC = xBC + (act_quant_fn(xBC, act_levels, -1) - xBC).detach()
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        if self.no_sftplus:
            def softplus_taylor_approx_3rd_order(x):
                return torch.log(torch.tensor(2.0)) + 0.5 * x + (1/8) * x**2 + (1/48) * x**3
            dt = dt + self.dt_bias.to(dtype=dt.dtype)
            dt = torch.where(dt < -2, torch.tensor(0.0), 
                     torch.where(dt > 2, dt, softplus_taylor_approx_3rd_order(dt)))
        else:
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)

        if self.taylor_exp:
            def taylor_approx_5th_order(x):
                return torch.tensor(1.0) + x + (1/2) * x**2 + (1/6) * x**3 + (1/24)*x**4 + (1/120)*x**5
            dA = torch.maximum(dt*A, -10000.0)
            scale = taylor_approx_5th_order(dA)
            scale = torch.minimum(torch.maximum(scale, 0.0), 1.0)
        else:
            scale = torch.exp(dt * A)  # (batch, nheads)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state = ssm_state * rearrange(scale, "b h -> b h 1 1") + dBx
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)

        y = rearrange(y, "b l h p -> b l (h p)")
        if self.quant:
            act_levels = 2**(self.quant_bits['act']-1)-1
            y = y + (act_quant_fn(y, act_levels, -1) - y).detach()

        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = y * self.act(z)  # (B D)

        if self.quant:
            act_levels = 2**(self.quant_bits['act']-1)-1
            y = y + (act_quant_fn(y, act_levels, -1) - y).detach()

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state
    

if __name__=="__main__":

    model = Mamba2(d_model=128, # Model dimension d_model
                    d_state=64,
                    quant_bits={'weights':4,'act':8})

    