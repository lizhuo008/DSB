# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx
from profiler.profiler import prof

@torch.compile()
def get_first_mask_idx(mask_blk: torch.Tensor) -> torch.Tensor:
    has_mask = mask_blk.any(dim=1)
    first_mask_idx = mask_blk.to(torch.long).argmax(dim=1)
    return first_mask_idx if has_mask else mask_blk.shape[1]

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index, _ = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index, _ = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe

@ torch.no_grad()
def generate_s(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    nfe = 0
    block_ptr = torch.full((x.shape[0], 1), prompt.shape[1], dtype=torch.long, device=x.device)
    low_conf_mask = torch.zeros_like(x, dtype=torch.bool)
    num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
    i = 0
    while True:
        nfe += 1
        mask_index = (x == mask_id)
        logits = model(x).logits
        mask_index[:, block_ptr + block_length:] = 0 # not consider low conf tokens yet
        if factor is None:
            x0, transfer_index, conf = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
        else:
            x0, transfer_index, conf = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        i += 1
        
        if block_ptr.min().item() < x.shape[1] - block_length:
            block_mask_index = x[:, block_ptr: block_ptr + block_length] == mask_id
            # skip low confidence tokens
            low_conf_mask = conf[:, block_ptr: block_ptr + block_length] > 0
            block_mask_index = block_mask_index & low_conf_mask if block_mask_index.sum(dim=-1, keepdim=True) <= 8 else block_mask_index
            whole_block = block_mask_index.sum(dim=-1, keepdim=True) == 0 # (B, 1) bool
            block_ptr = torch.clamp(block_ptr + block_mask_index.long().argmax(dim=-1, keepdim=True) + whole_block.long() * block_length, max=x.shape[1] - block_length)

        if (x[:, prompt.shape[1]:] == mask_id).sum() == 0:
            break
    
    print(f'nfe: {nfe}')
    return x, nfe

@ torch.no_grad()
def generate_i(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    nfe = 0
    block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length, dtype=torch.long, device=x.device)
    num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
    # print(f'num_transfer_tokens: {num_transfer_tokens}')
    i = 0
    while True:
        nfe += 1
        mask_index = (x == mask_id)
        logits = model(x).logits
        mask_index[:, block_end_ptr:] = 0 # not consider low conf tokens yet
        if factor is None:
            x0, transfer_index, _ = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
        else:
            x0, transfer_index, _ = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        i += 1

        if block_end_ptr.min().item() < x.shape[1]:
            extend_len = transfer_index.sum(dim=-1, keepdim=True)
            block_end_ptr = torch.clamp(block_end_ptr + extend_len, max=x.shape[1])

        if (x[:, prompt.shape[1]:] == mask_id).sum() == 0:
            break
    
    print(f'nfe: {nfe}')
    return x, nfe

@ torch.no_grad()
def generate_s_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, prefix_window=8, suffix_window=0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    twl = suffix_window if suffix_window > 0 else 0
    pwl = prefix_window if prefix_window > 0 else 0

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    B, L = x.shape

    nfe = 0
    # block_mask could act like the replace_position in the dual_cache
    # block_mask = torch.zeros_like(x, dtype=torch.bool)
    # block_mask[:, prompt.shape[1]: prompt.shape[1] + block_length + tbl] = True

    block_start_ptr = torch.full((x.shape[0], 1), prompt.shape[1], dtype=torch.long, device=x.device)
    # block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length + tbl, dtype=torch.long, device=x.device)
    num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
    # print(f'num_transfer_tokens: {num_transfer_tokens}')
    i = 0
    num_unmask = 0
    # print(f'L: {L}, gen_length: {gen_length}')
    while num_unmask < gen_length:
        blk_acc = 0
        prev_transfer_index = None
        cur_transfer_index = None
        # no s, e here, but a block_mask indicating the block to generate
        # a function like get_num_transfer_tokens need
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        global_mask_index = (x == mask_id)
        global_mask_index[:, block_start_ptr + block_length:] = 0

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, cur_transfer_index, _ = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, cur_transfer_index, _ = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )    
        
        num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
        x = torch.where(cur_transfer_index, x0, x)

        # new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
        # block_end_ptr = new_block_end_ptr

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start_ptr: block_start_ptr + block_length] = True

        blk_acc += num_acc.item()
        num_unmask += num_acc.item()
        
        twl = max(min(gen_length - num_unmask - block_length, twl), 0)
        # print(f'first step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
        # assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, cur_transfer_index.sum: {cur_transfer_index.sum().item()}'

        while True:
            
            cur_mask_blk = (x[:, block_start_ptr: block_start_ptr + block_length] == mask_id)
            cur_idx = get_first_mask_idx(cur_mask_blk)
            block_start_ptr = torch.clamp(block_start_ptr + cur_idx, max=x.shape[1] - block_length)
            
            if blk_acc >= block_length or num_unmask == gen_length:
                break

            prefix_window = max(cur_idx, pwl)

            input_x = x[:, block_start_ptr - prefix_window: block_start_ptr + block_length + twl]
            new_replace_position = torch.zeros_like(x, dtype=torch.bool)
            new_replace_position[:, block_start_ptr - prefix_window: block_start_ptr + block_length + twl] = True
            replace_position = new_replace_position
            out_blk = model(input_x, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
            logits_blk = out_blk.logits
            past_key_values = out_blk.past_key_values
            mask_blk = (input_x == mask_id)
            # print(f'mask_blk.shape: {mask_blk.shape}')
            if twl > 0:
                mask_blk[:, block_start_ptr + block_length:] = 0

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, cur_transfer_index, _ = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, input_x, quota_i, threshold
                )
            else:
                x0_blk, cur_transfer_index, _ = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, input_x, None, factor
                )

            num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
            
            blk_old = input_x
            blk_new = torch.where(cur_transfer_index, x0_blk, blk_old)
            x[:, block_start_ptr - prefix_window: block_start_ptr + block_length + twl] = blk_new

            blk_acc += num_acc.item()
            num_unmask += num_acc.item()
            twl = max(min(gen_length - num_unmask - block_length, twl), 0)
            # print(f'second step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
            # assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, mask_blk: {mask_blk}, remain: {(x == mask_id).sum().item()}, remain_idx: {(x == mask_id).nonzero(as_tuple=False)}'

            nfe += 1

            # if not num_unmask >= gen_length-block_length:
            #     print(f'num_unmask: {num_unmask}, block_mask.sum: {block_mask.sum()}')
            #     if prev_idx is not None:
            #         block_mask[rows, prev_idx] = block_mask[rows, prev_idx] & ~prev_transfer_index
            #     else:
            #         block_mask = block_mask & ~prev_transfer_index

            
            # prev_idx = (mask_blk).nonzero(as_tuple=False)
            # # print(f'prev_idx: {prev_idx.shape}, num_acc: {num_acc.item()}, mask_blk: {mask_blk}')
            # if prev_idx.shape[0] > 0:
            #     prev_idx = prev_idx[:, 1].view(B, -1)[:, 0]
            #     block_start_ptr = block_start_ptr + prev_idx
            #     # print(f'block_start_ptr: {block_start_ptr}, prev_idx: {prev_idx}')
            # else:
            #     block_start_ptr = block_start_ptr + prev_transfer_index.shape[1]
            

            # new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
            # block_end_ptr = new_block_end_ptr

            # new_replace_position = torch.zeros_like(x, dtype=torch.bool)
            # new_replace_position[:, block_start_ptr: block_start_ptr + block_length] = True
            # replace_position = new_replace_position

            # assert block_mask.sum() - num_acc.item() == block_length + tbl, f'block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}'
            # print(f'blk_acc: {blk_acc}, block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}')

    print(f'nfe: {nfe}')
    return x, nfe

# @ torch.no_grad()
# def generate_i_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              remasking='low_confidence', mask_id=126336, threshold=None, factor=None, prefix_window=8, suffix_window=0):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     twl = suffix_window if suffix_window > 0 else 0
#     pwl = prefix_window

#     x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     B, L = x.shape

#     nfe = 0
#     # block_mask could act like the replace_position in the dual_cache
#     # block_mask = torch.zeros_like(x, dtype=torch.bool)
#     # block_mask[:, prompt.shape[1]: prompt.shape[1] + block_length + tbl] = True
#     last_block_start_ptr = None
#     block_start_ptr = torch.full((x.shape[0], 1), prompt.shape[1], dtype=torch.long, device=x.device)
#     block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length, dtype=torch.long, device=x.device)
#     num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
#     # print(f'num_transfer_tokens: {num_transfer_tokens}')
#     i = 0
#     num_unmask = 0
#     replace_position = torch.zeros_like(x, dtype=torch.bool)
#     # print(f'L: {L}, gen_length: {gen_length}')
#     while num_unmask < gen_length:
#         # with prof.time_context("out loop"):
#         blk_acc = 0
#         cur_idx = 0
#         cur_transfer_index = None
#         last_block_start_ptr = block_start_ptr.clone()
#         # no s, e here, but a block_mask indicating the block to generate
#         # a function like get_num_transfer_tokens need
#         # with prof.time_context("out model_forward"):
#         out_full = model(x, use_cache=True)
#         past_key_values = out_full.past_key_values
#         nfe += 1

#         global_mask_index = (x == mask_id)
#         global_mask_index[:, block_end_ptr:] = 0

#         # with prof.time_context("out get_transfer_index"):
#         if factor is None:
#             quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
#             x0, cur_transfer_index, _ = get_transfer_index(
#                 out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
#             )
#         else:
#             x0, cur_transfer_index, _ = get_transfer_index_dynamic(
#                 out_full.logits, temperature, remasking, global_mask_index, x, None, factor
#             )    
        
#         num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
#         x = torch.where(cur_transfer_index, x0, x)

#         # new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#         # block_end_ptr = new_block_end_ptr

#         blk_acc += num_acc.item()
#         num_unmask += num_acc.item()
        
#         twl = max(min(gen_length - num_unmask - block_length, twl), 0)
#         # print(f'first step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
#         # assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, cur_transfer_index.sum: {cur_transfer_index.sum().item()}'

#         while True:
#             # cur_mask_blk = (x[:, block_start_ptr: block_end_ptr] == mask_id)
#             # cur_idx = (cur_mask_blk).nonzero(as_tuple=False)
#             # cur_idx = cur_idx[:, 1].view(B, -1)[:, 0] if cur_idx.shape[0] > 0 else cur_mask_blk.shape[1]
#             # with prof.time_context("update_block_ptr"):
#             block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])

#             cur_mask_blk = (x[:, block_start_ptr: block_end_ptr] == mask_id)
#             cur_idx = get_first_mask_idx(cur_mask_blk)
#             block_start_ptr = block_start_ptr + cur_idx
                
#             if blk_acc >= block_length or num_unmask == gen_length:
#                 break

#             # with prof.time_context("update_replace_position"):
#             # prefix_window = min(max(cur_idx, pwl), block_start_ptr - last_block_start_ptr)
#             prefix_window = cur_idx if nfe % 4 != 0 else block_start_ptr - last_block_start_ptr
#             # prefix_window = cur_idx + pwl

#             input_x = x[:, block_start_ptr - prefix_window: block_end_ptr + twl]
#             # new_replace_position = torch.zeros_like(x, dtype=torch.bool)
#             # new_replace_position[:, block_start_ptr - prefix_window: block_end_ptr + twl] = True
#             # replace_position = new_replace_position
#             replace_position.zero_()
#             replace_position[:, block_start_ptr - prefix_window: block_end_ptr + twl] = True

#             # with prof.time_context("model_forward"):
#             out_blk = model(input_x, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
#             nfe += 1
#             logits_blk = out_blk.logits
#             past_key_values = out_blk.past_key_values
#             mask_blk = (input_x == mask_id)
#             # print(f'mask_blk.shape: {mask_blk.shape}')
#             if twl > 0:
#                 mask_blk[:, block_end_ptr:] = 0

#             # with prof.time_context("get_transfer_index"):
#             if factor is None:
#                 quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
#                 x0_blk, cur_transfer_index, _ = get_transfer_index(
#                     logits_blk, temperature, remasking, mask_blk, input_x, quota_i, threshold
#                 )
#             else:
#                 x0_blk, cur_transfer_index, _ = get_transfer_index_dynamic(
#                     logits_blk, temperature, remasking, mask_blk, input_x, None, factor
#                 )
#             # with prof.time_context("others"):
#             num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
            
#             blk_old = input_x
#             blk_new = torch.where(cur_transfer_index, x0_blk, blk_old)
#             x = torch.cat([x[:, :block_start_ptr - prefix_window], blk_new, x[:, block_end_ptr + twl:]], dim=1)

#             blk_acc += num_acc.item()
#             num_unmask += num_acc.item()
#             twl = max(min(gen_length - num_unmask - block_length, twl), 0)
#             # print(f'second step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
#             # assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, remain: {(x == mask_id).sum().item()}, remain_idx: {(x == mask_id).nonzero(as_tuple=False)}'

                

#             # new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#             # block_end_ptr = new_block_end_ptr

#     print(f'nfe: {nfe}')
#     return x, nfe

@ torch.no_grad()
def generate_i_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, prefix_window=8, suffix_window=0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    twl = suffix_window if suffix_window > 0 else 0
    pwl = prefix_window if prefix_window > 0 else 0

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    B, L = x.shape

    nfe = 0
    # block_mask could act like the replace_position in the dual_cache
    # block_mask = torch.zeros_like(x, dtype=torch.bool)
    # block_mask[:, prompt.shape[1]: prompt.shape[1] + block_length + tbl] = True

    block_start_ptr = torch.full((x.shape[0], 1), prompt.shape[1], dtype=torch.long, device=x.device)
    block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length, dtype=torch.long, device=x.device)
    num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
    # print(f'num_transfer_tokens: {num_transfer_tokens}')
    i = 0
    num_unmask = 0
    replace_position = torch.zeros_like(x, dtype=torch.bool)
    # print(f'L: {L}, gen_length: {gen_length}')
    while num_unmask < gen_length:
        # with prof.time_context("out loop"):
        blk_acc = 0
        cur_idx = 0
        cur_transfer_index = None
        # no s, e here, but a block_mask indicating the block to generate
        # a function like get_num_transfer_tokens need
        # with prof.time_context("out model_forward"):
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        global_mask_index = (x == mask_id)
        global_mask_index[:, block_end_ptr:] = 0

        # with prof.time_context("out get_transfer_index"):
        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, cur_transfer_index, _ = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, cur_transfer_index, _ = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )    
        
        num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
        x = torch.where(cur_transfer_index, x0, x)

        # new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
        # block_end_ptr = new_block_end_ptr

        blk_acc += num_acc.item()
        num_unmask += num_acc.item()
        
        twl = max(min(gen_length - num_unmask - block_length, twl), 0)
        # print(f'first step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
        # assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, cur_transfer_index.sum: {cur_transfer_index.sum().item()}'

        while True:
            # cur_mask_blk = (x[:, block_start_ptr: block_end_ptr] == mask_id)
            # cur_idx = (cur_mask_blk).nonzero(as_tuple=False)
            # cur_idx = cur_idx[:, 1].view(B, -1)[:, 0] if cur_idx.shape[0] > 0 else cur_mask_blk.shape[1]
            # with prof.time_context("update_block_ptr"):
            block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])

            cur_mask_blk = (x[:, block_start_ptr: block_end_ptr] == mask_id)
            cur_idx = get_first_mask_idx(cur_mask_blk)
            block_start_ptr = block_start_ptr + cur_idx
                
            if blk_acc >= block_length or num_unmask == gen_length:
                break

            # with prof.time_context("update_replace_position"):
            prefix_window = max(cur_idx, pwl)
            # prefix_window = cur_idx + pwl

            input_x = x[:, block_start_ptr - prefix_window: block_end_ptr + twl]
            # new_replace_position = torch.zeros_like(x, dtype=torch.bool)
            # new_replace_position[:, block_start_ptr - prefix_window: block_end_ptr + twl] = True
            # replace_position = new_replace_position
            replace_position.zero_()
            replace_position[:, block_start_ptr - prefix_window: block_end_ptr + twl] = True

            # with prof.time_context("model_forward"):
            out_blk = model(input_x, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
            nfe += 1
            logits_blk = out_blk.logits
            past_key_values = out_blk.past_key_values
            mask_blk = (input_x == mask_id)
            # print(f'mask_blk.shape: {mask_blk.shape}')
            if twl > 0:
                mask_blk[:, block_end_ptr:] = 0

            # with prof.time_context("get_transfer_index"):
            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, cur_transfer_index, _ = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, input_x, quota_i, threshold
                )
            else:
                x0_blk, cur_transfer_index, _ = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, input_x, None, factor
                )
            # with prof.time_context("others"):
            num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
            
            blk_old = input_x
            blk_new = torch.where(cur_transfer_index, x0_blk, blk_old)
            x = torch.cat([x[:, :block_start_ptr - prefix_window], blk_new, x[:, block_end_ptr + twl:]], dim=1)

            blk_acc += num_acc.item()
            num_unmask += num_acc.item()
            twl = max(min(gen_length - num_unmask - block_length, twl), 0)
            # print(f'second step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
            # assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, remain: {(x == mask_id).sum().item()}, remain_idx: {(x == mask_id).nonzero(as_tuple=False)}'

                

            # new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
            # block_end_ptr = new_block_end_ptr

    print(f'nfe: {nfe}')
    return x, nfe

# @ torch.no_grad()
# def generate_i_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              remasking='low_confidence', mask_id=126336, threshold=None, factor=None, tail_block_length=0):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     tbl = tail_block_length if tail_block_length > 0 else 0

#     x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     B, L = x.shape

#     nfe = 0
#     # block_mask could act like the replace_position in the dual_cache
#     # block_mask = torch.zeros_like(x, dtype=torch.bool)
#     # block_mask[:, prompt.shape[1]: prompt.shape[1] + block_length + tbl] = True

#     block_start_ptr = torch.full((x.shape[0], 1), prompt.shape[1], dtype=torch.long, device=x.device)
#     block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length + tbl, dtype=torch.long, device=x.device)
#     num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
#     # print(f'num_transfer_tokens: {num_transfer_tokens}')
#     i = 0
#     num_unmask = 0
#     # print(f'L: {L}, gen_length: {gen_length}')
#     while num_unmask < gen_length:
#         blk_acc = 0
#         prev_transfer_index = None
#         cur_transfer_index = None
#         # no s, e here, but a block_mask indicating the block to generate
#         # a function like get_num_transfer_tokens need
#         out_full = model(x, use_cache=True)
#         past_key_values = out_full.past_key_values
#         nfe += 1

#         global_mask_index = (x == mask_id)
#         global_mask_index[:, block_end_ptr - tbl:] = 0

#         if factor is None:
#             quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
#             x0, cur_transfer_index, _ = get_transfer_index(
#                 out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
#             )
#         else:
#             x0, cur_transfer_index, _ = get_transfer_index_dynamic(
#                 out_full.logits, temperature, remasking, global_mask_index, x, None, factor
#             )    
        
#         num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
#         x = torch.where(cur_transfer_index, x0, x)

#         new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#         block_end_ptr = new_block_end_ptr

#         replace_position = torch.zeros_like(x, dtype=torch.bool)
#         replace_position[:, block_start_ptr: block_end_ptr] = True

#         blk_acc += num_acc.item()
#         num_unmask += num_acc.item()
        
#         tbl = max(min(gen_length - num_unmask - block_length, tbl), 0)
#         # print(f'first step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
#         assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, cur_transfer_index.sum: {cur_transfer_index.sum().item()}'

#         while True:
#             if blk_acc >= block_length or num_unmask == gen_length:
#                 # print(f'blk_acc: {blk_acc}, num_unmask: {num_unmask}, tbl: {tbl}')
#                 if num_unmask != gen_length:
#                     cur_mask_blk = (x[:, block_start_ptr: block_end_ptr] == mask_id)
#                     cur_idx = (cur_mask_blk).nonzero(as_tuple=False)
#                     if cur_idx.shape[0] > 0:
#                         cur_idx = cur_idx[:, 1].view(B, -1)[:, 0]
#                         block_start_ptr = block_start_ptr + cur_idx
#                     else:
#                         block_start_ptr = block_start_ptr + cur_transfer_index.shape[1]
#                 break

#             prev_transfer_index = cur_transfer_index

#             input_x = x[:, block_start_ptr: block_end_ptr]
#             out_blk = model(input_x, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
#             logits_blk = out_blk.logits
#             past_key_values = out_blk.past_key_values
#             mask_blk = (input_x == mask_id)
#             # print(f'mask_blk.shape: {mask_blk.shape}')
#             if tbl > 0:
#                 mask_blk[:, -tbl:] = 0

#             if factor is None:
#                 quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
#                 x0_blk, cur_transfer_index, _ = get_transfer_index(
#                     logits_blk, temperature, remasking, mask_blk, input_x, quota_i, threshold
#                 )
#             else:
#                 x0_blk, cur_transfer_index, _ = get_transfer_index_dynamic(
#                     logits_blk, temperature, remasking, mask_blk, input_x, None, factor
#                 )

#             num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
            
#             blk_old = input_x
#             blk_new = torch.where(cur_transfer_index, x0_blk, blk_old)
#             x[:, block_start_ptr: block_end_ptr] = blk_new

#             blk_acc += num_acc.item()
#             num_unmask += num_acc.item()
#             tbl = max(min(gen_length - num_unmask - block_length, tbl), 0)
#             # print(f'second step: num_unmask: {num_unmask}, blk_acc: {blk_acc}, tbl: {tbl}')
#             assert num_acc.item() > 0, f'num_acc: {num_acc.item()}, remain: {(x == mask_id).sum().item()}, remain_idx: {(x == mask_id).nonzero(as_tuple=False)}'

#             nfe += 1

#             # if not num_unmask >= gen_length-block_length:
#             #     print(f'num_unmask: {num_unmask}, block_mask.sum: {block_mask.sum()}')
#             #     if prev_idx is not None:
#             #         block_mask[rows, prev_idx] = block_mask[rows, prev_idx] & ~prev_transfer_index
#             #     else:
#             #         block_mask = block_mask & ~prev_transfer_index

            
#             prev_idx = (mask_blk).nonzero(as_tuple=False)
#             # print(f'prev_idx: {prev_idx.shape}, num_acc: {num_acc.item()}, mask_blk: {mask_blk}')
#             if prev_idx.shape[0] > 0:
#                 prev_idx = prev_idx[:, 1].view(B, -1)[:, 0]
#                 block_start_ptr = block_start_ptr + max(prev_idx - 4, 0)
#                 # print(f'block_start_ptr: {block_start_ptr}, prev_idx: {prev_idx}')
#             else:
#                 block_start_ptr = block_start_ptr + max(prev_transfer_index.shape[1] - 4, 0)
            

#             new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#             block_end_ptr = new_block_end_ptr

#             new_replace_position = torch.zeros_like(x, dtype=torch.bool)
#             new_replace_position[:, block_start_ptr: block_end_ptr] = True
#             replace_position = new_replace_position

#             # assert block_mask.sum() - num_acc.item() == block_length + tbl, f'block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}'
#             # print(f'blk_acc: {blk_acc}, block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}')

#     print(f'nfe: {nfe}')
#     return x, nfe

# @ torch.no_grad()
# # discrete block
# def generate_i_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              remasking='low_confidence', mask_id=126336, threshold=None, factor=None, tail_block_length=0):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     tbl = tail_block_length if tail_block_length > 0 else 0

#     x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     B, L = x.shape

#     nfe = 0
#     # block_mask could act like the replace_position in the dual_cache
#     block_mask = torch.zeros_like(x, dtype=torch.bool)
#     block_mask[:, prompt.shape[1]: prompt.shape[1] + block_length + tbl] = True


#     block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length + tbl, dtype=torch.long, device=x.device)
#     num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
#     # print(f'num_transfer_tokens: {num_transfer_tokens}')
#     i = 0
#     num_unmask = 0
#     # print(f'L: {L}, gen_length: {gen_length}')
#     while num_unmask < gen_length:
#         blk_acc = 0
#         prev_idx = None
#         cur_idx = None
#         prev_transfer_index = None
#         cur_transfer_index = None
#         # no s, e here, but a block_mask indicating the block to generate
#         # a function like get_num_transfer_tokens need
#         out_full = model(x, use_cache=True)
#         past_key_values = out_full.past_key_values
#         nfe += 1

#         global_mask_index = (x == mask_id)
#         global_mask_index[:, block_end_ptr - tbl:] = 0

#         if factor is None:
#             quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
#             x0, cur_transfer_index, _ = get_transfer_index(
#                 out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
#             )
#         else:
#             x0, cur_transfer_index, _ = get_transfer_index_dynamic(
#                 out_full.logits, temperature, remasking, global_mask_index, x, None, factor, max_transfer_tokens=block_length-blk_acc
#             )    
        
#         num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
#         x = torch.where(cur_transfer_index, x0, x)

#         new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#         block_mask[:, block_end_ptr: new_block_end_ptr] = True
#         block_end_ptr = new_block_end_ptr

#         blk_acc += num_acc.item()
#         num_unmask += num_acc.item()
#         tbl = max(min(gen_length - num_unmask - block_length, tbl), 0)

#         while True:
#             prev_transfer_index = cur_transfer_index
#             prev_idx = cur_idx

#             cur_idx = torch.nonzero(block_mask, as_tuple=False)
#             cur_idx = cur_idx[:, 1].view(B, -1)

#             rows = torch.arange(B, device=x.device)[:, None]
#             input_x = x[rows, cur_idx]
#             out_blk = model(input_x, past_key_values=past_key_values, use_cache=True, replace_position=block_mask)
#             logits_blk = out_blk.logits
#             past_key_values = out_blk.past_key_values
#             mask_blk = (input_x == mask_id)
#             if tbl > 0:
#                 mask_blk[:, -tbl:] = 0

#             if factor is None:
#                 quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
#                 x0_blk, cur_transfer_index, _ = get_transfer_index(
#                     logits_blk, temperature, remasking, mask_blk, input_x, quota_i, threshold
#                 )
#             else:
#                 x0_blk, cur_transfer_index, _ = get_transfer_index_dynamic(
#                     logits_blk, temperature, remasking, mask_blk, input_x, None, factor, max_transfer_tokens=block_length-blk_acc
#                 )

#             num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
#             blk_old = input_x
#             blk_new = torch.where(cur_transfer_index, x0_blk, blk_old)
#             x[rows, cur_idx] = blk_new

#             blk_acc += num_acc.item()
#             num_unmask += num_acc.item()
#             tbl = max(min(gen_length - num_unmask - block_length, tbl), 0)

#             nfe += 1

#             # if not num_unmask >= gen_length-block_length:
#             #     print(f'num_unmask: {num_unmask}, block_mask.sum: {block_mask.sum()}')
#             #     if prev_idx is not None:
#             #         block_mask[rows, prev_idx] = block_mask[rows, prev_idx] & ~prev_transfer_index
#             #     else:
#             #         block_mask = block_mask & ~prev_transfer_index

#             if prev_transfer_index is not None:
#                 block_mask[rows, prev_idx] = block_mask[rows, prev_idx] & ~prev_transfer_index
            

#             new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#             block_mask[:, block_end_ptr: new_block_end_ptr] = True
#             block_end_ptr = new_block_end_ptr

#             # assert block_mask.sum() - num_acc.item() == block_length + tbl, f'block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}'
#             # print(f'blk_acc: {blk_acc}, block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}')

#             if blk_acc >= block_length or num_unmask == gen_length:
#                 # print(f'blk_acc: {blk_acc}, num_unmask: {num_unmask}, tbl: {tbl}')
#                 block_mask[rows, cur_idx] = block_mask[rows, cur_idx] & ~cur_transfer_index
#                 break

#     print(f'nfe: {nfe}')
#     return x, nfe

# @ torch.no_grad()
# # discrete block
# def generate_i_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              remasking='low_confidence', mask_id=126336, threshold=None, factor=None, tail_block_length=0):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     tbl = tail_block_length if tail_block_length > 0 else 0

#     x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     B, L = x.shape

#     nfe = 0
#     # block_mask could act like the replace_position in the dual_cache
#     block_mask = torch.zeros_like(x, dtype=torch.bool)
#     block_mask[:, prompt.shape[1]: prompt.shape[1] + block_length + tbl] = True


#     last_block_start_ptr = None
#     block_end_ptr = torch.full((x.shape[0], 1), prompt.shape[1] + block_length + tbl, dtype=torch.long, device=x.device)
#     num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps).to(torch.long)  # (B, steps)
#     # cur_idx = torch.arange(prompt.shape[1], prompt.shape[1] + block_length + tbl, device=x.device)
#     # print(f'num_transfer_tokens: {num_transfer_tokens}')
#     i = 0
#     num_unmask = 0
#     rows = torch.arange(B, device=x.device)[:, None]
#     # print(f'L: {L}, gen_length: {gen_length}')
#     while num_unmask < gen_length:
#         blk_acc = 0
#         cur_idx = torch.arange(0, L, device=x.device)
#         cur_transfer_index = None
#         last_block_start_ptr = get_first_mask_idx(block_mask)
#         # no s, e here, but a block_mask indicating the block to generate
#         # a function like get_num_transfer_tokens need
#         out_full = model(x, use_cache=True)
#         past_key_values = out_full.past_key_values
#         nfe += 1

#         # global_mask_index = (x == mask_id)
#         # global_mask_index[:, block_end_ptr - tbl:] = 0

#         if factor is None:
#             quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
#             x0, cur_transfer_index, _ = get_transfer_index(
#                 out_full.logits, temperature, remasking, block_mask, x, quota0, threshold
#             )
#         else:
#             x0, cur_transfer_index, _ = get_transfer_index_dynamic(
#                 out_full.logits, temperature, remasking, block_mask, x, None, factor
#             )    
        
#         num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
#         x = torch.where(cur_transfer_index, x0, x)

#         blk_acc += num_acc.item()
#         num_unmask += num_acc.item()
#         tbl = max(min(gen_length - num_unmask - block_length, tbl), 0)

#         while True:
#             new_block_end_ptr = torch.clamp(block_end_ptr + num_acc, max=x.shape[1])
#             block_mask[:, block_end_ptr: new_block_end_ptr] = True
#             block_end_ptr = new_block_end_ptr

#             if blk_acc >= block_length or num_unmask == gen_length:
#                 # print(f'blk_acc: {blk_acc}, num_unmask: {num_unmask}, tbl: {tbl}')
#                 block_mask[rows, cur_idx] = block_mask[rows, cur_idx] & ~cur_transfer_index
#                 break

#             prev_transfer_index = cur_transfer_index
#             prev_idx = cur_idx
            
#             if nfe % 2 != 0:
#                 cur_idx = torch.nonzero(block_mask, as_tuple=False)
#                 cur_idx = cur_idx[:, 1].view(B, -1)
#                 replace_position = block_mask
#             else:
#                 # print(f'last_block_start_ptr: {last_block_start_ptr}, block_end_ptr: {block_end_ptr},unmask_num: {num_unmask}')
#                 cur_idx = torch.arange(last_block_start_ptr.item(), block_end_ptr.item(), device=x.device)
#                 replace_position = torch.zeros_like(x, dtype=torch.bool)
#                 replace_position[:, last_block_start_ptr.item(): block_end_ptr.item()] = True

#             input_x = x[rows, cur_idx]
#             out_blk = model(input_x, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
#             logits_blk = out_blk.logits
#             past_key_values = out_blk.past_key_values
#             mask_blk = (input_x == mask_id)
#             if tbl > 0:
#                 mask_blk[:, -tbl:] = 0

#             if factor is None:
#                 quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
#                 x0_blk, cur_transfer_index, _ = get_transfer_index(
#                     logits_blk, temperature, remasking, mask_blk, input_x, quota_i, threshold
#                 )
#             else:
#                 x0_blk, cur_transfer_index, _ = get_transfer_index_dynamic(
#                     logits_blk, temperature, remasking, mask_blk, input_x, None, factor
#                 )

#             num_acc = cur_transfer_index.sum(dim=-1, keepdim=True)
#             blk_old = input_x
#             blk_new = torch.where(cur_transfer_index, x0_blk, blk_old)
#             x[rows, cur_idx] = blk_new

#             blk_acc += num_acc.item()
#             num_unmask += num_acc.item()
#             tbl = max(min(gen_length - num_unmask - block_length, tbl), 0)

#             nfe += 1

#             # if not num_unmask >= gen_length-block_length:
#             #     print(f'num_unmask: {num_unmask}, block_mask.sum: {block_mask.sum()}')
#             #     if prev_idx is not None:
#             #         block_mask[rows, prev_idx] = block_mask[rows, prev_idx] & ~prev_transfer_index
#             #     else:
#             #         block_mask = block_mask & ~prev_transfer_index

#             if prev_transfer_index is not None:
#                 block_mask[rows, prev_idx] = block_mask[rows, prev_idx] & ~prev_transfer_index
            

            

#             # assert block_mask.sum() - num_acc.item() == block_length + tbl, f'block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}'
#             # print(f'blk_acc: {blk_acc}, block_mask.sum: {block_mask.sum()}, num_acc: {num_acc.item()}')

            

#     print(f'nfe: {nfe}')
#     return x, nfe


@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index, _ = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index, _ = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk, _ = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk, _ = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe



def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
    max_transfer_tokens: int = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        if transfer_index.sum() > max_transfer_tokens:
                # get top max_accept tokens
            _, indices = torch.topk(confidence, k=max_transfer_tokens, largest=True)
            transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
            transfer_index.view(-1)[indices] = True

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index, x0_p

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index, x0_p

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1, max_transfer_tokens=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        if max_transfer_tokens is not None:
            top_i = min(top_i, max_transfer_tokens)

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index, x0_p

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
