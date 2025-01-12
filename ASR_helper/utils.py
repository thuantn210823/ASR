from typing import Callable

import torch

import os
import collections

def avg_ckpt(ckpt_directory: str):
    total_ckpt = {'state_dict': collections.defaultdict(list)}
    ckpt_files = os.listdir(ckpt_directory)
    for ckpt_file in ckpt_files:
        ckpt = torch.load(os.path.join(ckpt_directory, ckpt_file), map_location = 'cpu', weights_only = False)
        state_dict = ckpt['state_dict']
        for key, value in state_dict.items():
            total_ckpt['state_dict'][key].append(value)
    for key, value in total_ckpt['state_dict'].items():
        value = torch.stack(value).mean(dim = 0)
        total_ckpt['state_dict'][key] = value
    return total_ckpt

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False).type(torch.bool)
    return mask

def create_tgt_mask(tgt, 
                    pad_idx: int = 0):
    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == pad_idx)
    return tgt_mask.to(tgt.device), tgt_padding_mask.to(tgt.device)

def transcribe(model: Callable,
               tokenizer: Callable,
               src: torch.Tensor,
               src_lengths: torch.Tensor,
               max_tgt_lengths: int):
    model.eval()
    texts = []
    token_outs = model.generate(src, src_lengths, max_tgt_lengths)
    for token_out in token_outs:
        texts.append(tokenizer.itos(token_out))
    return texts
