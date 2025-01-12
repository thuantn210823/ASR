from typing import Tuple

import torch
from torch import nn

from ASR_helper.Conformer import ConformerEncoderLayer, ConformerEncoder
from ASR_helper.RNNT import _TimeReduction, _Predictor, _Joiner, RNNT

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
                 conformer_dropout: float,
                 *args, **kwargs) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = nn.Linear(input_dim*time_reduction_stride, conformer_input_dim)
        conformerencoderlayer = ConformerEncoderLayer(input_dim = conformer_input_dim,
                                                      ffn_dim = conformer_ffn_dim,
                                                      num_heads = conformer_num_heads,
                                                      kernel_size = conformer_depthwise_conv_kernel_size,
                                                      dropout = conformer_dropout,
                                                      *args, **kwargs)
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
        x, lengths, _ = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths

class ConformerTransducer(RNNT):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 time_reduction_stride: int,
                 conformer_input_dim: int,
                 conformer_ffn_dim: int,
                 conformer_num_layers: int,
                 conformer_num_heads: int,
                 conformer_depthwise_conv_kernel_size: int,
                 conformer_dropout: float,
                 num_symbols: int,
                 symbol_embedding_dim: int,
                 num_lstm_layers: int,
                 lstm_hidden_dim: int,
                 lstm_layer_norm: bool = False,
                 lstm_layer_norm_epsilon: float = 1e-5,
                 lstm_dropout: float = 0.0,
                 joiner_activation: str = 'relu',
                 *args, **kwargs) -> None:
        transcriber = _ConformerEncoder(input_dim,
                                        output_dim,
                                        time_reduction_stride,
                                        conformer_input_dim,
                                        conformer_ffn_dim,
                                        conformer_num_layers,
                                        conformer_num_heads,
                                        conformer_depthwise_conv_kernel_size,
                                        conformer_dropout,
                                        *args, **kwargs)
        predictor = _Predictor(num_symbols,
                                    output_dim,
                                    symbol_embedding_dim,
                                    num_lstm_layers,
                                    lstm_hidden_dim,
                                    lstm_layer_norm,
                                    lstm_layer_norm_epsilon,
                                    lstm_dropout)
        joiner = _Joiner(output_dim, num_symbols, activation = joiner_activation)
        super().__init__(transcriber,
                         predictor,
                         joiner)

    def generate(self, 
                 input: torch.Tensor, 
                 input_lengths: torch.Tensor,
                 max_tgt_lengths: torch.Tensor = 400,
                 start_symbol_idx: int = 2,
                 end_symbol_idx: int = 3,
                 pad_symbol_idx: int = 0):
        """
        Greedy Search
        """
        y_batch = []
        B = len(input)
        enc_out, enc_lengths = self.transcriber(input, input_lengths)
        for b in range(B):
            t = 0; u = 0;
            y = [start_symbol_idx]
            predictor_state = None
            while t < enc_lengths[b] and u < max_tgt_lengths:
                predictor_in = torch.tensor([y[-1]], device = input.device).reshape(1, 1)
                predictor_out, _, predictor_state = self.predictor(predictor_in, None, predictor_state)
                transcriber_out = enc_out[b, t].reshape(1, 1, -1)
                joiner_out, _, _ = self.joiner(transcriber_out,
                                         None,
                                         predictor_out,
                                         None)
                argmax = joiner_out.max(-1)[1].item()
                if argmax == pad_symbol_idx:
                    t += 1
                elif argmax == end_symbol_idx:
                    break
                else:
                    u += 1
                    y.append(argmax)
            y_batch.append(y[1:])
        return y_batch