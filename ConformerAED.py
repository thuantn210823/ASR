from typing import Optional, Tuple

import torch
from torch import nn

from ASR_helper.Embedding import TokenEmbedding, PositionalEncoding
from ASR_helper.Conformer import ConformerEncoderLayer, ConformerEncoder
from ASR_helper.Transformer import CustomTransformerDecoder, CustomTransformerDecoderLayer
from ASR_helper.RNNT import _TimeReduction
from ASR_helper.utils import generate_square_subsequent_mask

class _ConformerEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 time_reduction_stride: int,
                 conformer_input_dim: int,
                 conformer_ffn_dim: int,
                 conformer_num_layers: int,
                 conformer_num_heads: int,
                 conformer_depthwise_conv_kernel_size: int,
                 conformer_dropout: float) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = nn.Linear(input_dim*time_reduction_stride, conformer_input_dim)
        conformerencoderlayer = ConformerEncoderLayer(input_dim = conformer_input_dim,
                                                      ffn_dim = conformer_ffn_dim,
                                                      num_heads = conformer_num_heads,
                                                      kernel_size = conformer_depthwise_conv_kernel_size,
                                                      dropout = conformer_dropout)
        self.conformer = ConformerEncoder(conformerencoderlayer,
                                          num_layers = conformer_num_layers)
        self.output_linear = nn.Linear(conformer_input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self,
                input: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: (N, T, D)
        length: (N,)
        """
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, _, key_padding_mask = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, key_padding_mask

class _TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_blks: int,
                 dropout = 0.1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size
        self.tgt_tok_emb = TokenEmbedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        decoder_layer = CustomTransformerDecoderLayer(d_model = num_hiddens,
                                                      n_head = num_heads,
                                                      dim_feedforward = ffn_num_hiddens,
                                                      dropout = dropout,
                                                      rel_attn = False)
        self.transformer_decoder = CustomTransformerDecoder(decoder_layer, num_blks)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        tgt_emb = self.pos_encoding(self.tgt_tok_emb(tgt))
        return self.transformer_decoder(tgt_emb, memory, tgt_mask, tgt_key_padding_mask,
                                        memory_key_padding_mask)

class ConformerAED(nn.Module):
    def __init__(self,
                 input_dim: int,
                 time_reduction_stride: int,
                 conformer_input_dim: int,
                 conformer_output_dim: int,
                 conformer_ffn_dim: int,
                 conformer_num_layers: int,
                 conformer_num_heads: int,
                 conformer_depthwise_conv_kernel_size: int,
                 conformer_dropout: float,
                 vocab_size: int,
                 decoder_input_dim: int,
                 decoder_ffn_dim: int,
                 decoder_num_layers: int,
                 decoder_num_heads: int,
                 decoder_dropout: float,
                 *args, **kwargs) -> None:
        super().__init__()
        self.conformer_output_dim = conformer_output_dim
        self.decoder_input_dim = decoder_input_dim
        self.encoder = _ConformerEncoder(input_dim,
                                         conformer_output_dim,
                                         time_reduction_stride,
                                         conformer_input_dim,
                                         conformer_ffn_dim,
                                         conformer_num_layers,
                                         conformer_num_heads,
                                         conformer_depthwise_conv_kernel_size,
                                         conformer_dropout,
                                         *args, **kwargs)
        self.decoder = _TransformerDecoder(vocab_size,
                                           decoder_input_dim,
                                           decoder_ffn_dim,
                                           decoder_num_heads,
                                           decoder_num_layers,
                                           decoder_dropout)
        if conformer_output_dim != decoder_input_dim:
            self.cross_linear = nn.Linear(conformer_output_dim, decoder_input_dim)
        self.classifier = nn.Linear(decoder_input_dim, vocab_size)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        src: shape (N, Ts, F)
        tgt: shape (N, Tt)
        """
        memory, memory_key_padding_mask  = self.encoder(src,
                                                        src_lengths)
        if self.conformer_output_dim != self.decoder_input_dim:
            memory = self.cross_linear(memory)
        outs = self.decoder(tgt, memory,
                            tgt_mask = tgt_mask,
                            tgt_key_padding_mask = tgt_padding_mask,
                            memory_key_padding_mask = memory_key_padding_mask)
        return self.classifier(outs)
    
    def generate(self,
                 src: torch.Tensor,
                 src_lengths: Optional[torch.Tensor],
                 max_tgt_lengths: int = 400,
                 start_symbol_idx: int = 2,
                 end_symbol_idx: int = 3):
        """
        Greedy Search
        """
        y_batch = []
        B = len(src)
        memory, memory_key_padding_mask = self.encoder(src, src_lengths)
        if self.conformer_output_dim != self.decoder_input_dim:
            memory = self.cross_linear(memory)
        for b in range(B):
            dec_input = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(src)
            dec_logits = []
            for i in range(max_tgt_lengths-1):
                tgt_mask = generate_square_subsequent_mask(dec_input.shape[1])
                out = self.decoder(dec_input, memory[b].unsqueeze(0), tgt_mask, memory_key_padding_mask = memory_key_padding_mask[b].unsqueeze(0))
                out = out.transpose(1, 2).contiguous()
                prob = self.classifier(out[:, :, -1])
                _, next_word = torch.max(prob, dim = -1)
                if next_word == end_symbol_idx:
                    break
                dec_logits.append(next_word.item())
                dec_input = torch.cat([dec_input, torch.ones(1, 1).type_as(src.data).fill_(next_word.item())], dim = 1)
            y_batch.append(dec_logits)
        return y_batch