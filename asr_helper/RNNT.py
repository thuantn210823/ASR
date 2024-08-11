from typing import Optional, Callable, Tuple, List

import torch
from torch import nn
import torchaudio

class _TimeReduction(nn.Module):
    """
    Coalesces frames along time dimension into a fewer number of frames with higher feature dimensionality

    Args:
        stride (int): number of frames to merge for each output frame
    """
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self,
                input: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input: (B, T, D)
        lengths: (B,)

        Returns:
            (B, T//stride, D*stride)
            output_lengths
        """
        B, T, D = input.shape
        num_frames = T - (T%self.stride)
        input = input[:, :num_frames, :]
        lengths = lengths.div(self.stride, rounding_mode = 'trunc')
        T_max = num_frames//self.stride

        output = input.reshape(B, T_max, D*self.stride).contiguous()
        return output, lengths
    
class _CustomLSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 layer_norm: bool = False,
                 layer_norm_epsilon: float = 1e-5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.x2g = nn.Linear(input_dim, 4*hidden_dim, bias = (not layer_norm))
        self.p2g = nn.Linear(hidden_dim, 4*hidden_dim, bias = False)
        if layer_norm:
            self.c_norm = nn.LayerNorm(hidden_dim, eps = layer_norm_epsilon)
            self.g_norm = nn.LayerNorm(4*hidden_dim, eps = layer_norm_epsilon)
        else:
            self.c_norm = nn.Identity()
            self.g_norm = nn.Identity()

    def forward(self,
                input: torch.Tensor,
                state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        input: (T, B, D)
        """
        if state is None:
            B = input.size(1)
            h = torch.zeros(B, self.hidden_dim, device = input.device, dtype = input.dtype)
            c = torch.zeros(B, self.hidden_dim, device = input.device, dtype = input.dtype)
        else:
            h, c = state
        
        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):           # Removes a tensor dimension -> seq
            gates = gates + self.p2g(h)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            c = forget_gate*c + input_gate*cell_gate
            c = self.c_norm(c)
            h = output_gate*c.tanh()
            outputs.append(h)
        
        output = torch.stack(outputs, dim = 0)
        state = [h, c]

        return output, state

class _Predictor_Kiss(nn.Module):
    def __init__(self,
                 num_symbols: int,
                 output_dim: int,
                 symbol_embedding_dim: int,
                 num_lstm_layers: int,
                 lstm_hidden_dim: int,
                 lstm_dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = nn.LayerNorm(symbol_embedding_dim)
        self.lstm = nn.LSTM(symbol_embedding_dim,
                            lstm_hidden_dim,
                            num_lstm_layers,
                            batch_first = True,
                            dropout = lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim, output_dim)
        self.output_layer_norm = nn.LayerNorm(output_dim)


    def forward(self,
                input: torch.Tensor,
                lengths: torch.Tensor,
                state: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]:
        embedding = self.embedding(input)
        lstm_out = self.input_layer_norm(embedding)
        lstm_out, state_out = self.lstm(lstm_out, state)
        out = self.linear(lstm_out)
        return self.output_layer_norm(out), lengths, state_out

class _Predictor(nn.Module):
    def __init__(self,
                 num_symbols: int,
                 output_dim: int,
                 symbol_embedding_dim: int,
                 num_lstm_layers: int,
                 lstm_hidden_dim: int,
                 lstm_layer_norm: bool = False,
                 lstm_layer_norm_epsilon: float = 1e-5,
                 lstm_dropout: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = nn.ModuleList([
            _CustomLSTM(
                symbol_embedding_dim if idx == 0 else lstm_hidden_dim,
                lstm_hidden_dim,
                layer_norm = lstm_layer_norm,
                layer_norm_epsilon = lstm_layer_norm_epsilon
            )
            for idx in range(num_lstm_layers)
        ])
        self.dropout = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim, output_dim)
        self.output_layer_norm = nn.LayerNorm(output_dim)


    def forward(self,
                input: torch.Tensor,
                lengths: torch.Tensor,
                state: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]:
        embedding = self.embedding(input)
        lstm_out = self.input_layer_norm(embedding).transpose(0, 1).contiguous()
        h_out: List[torch.Tensor] = []
        c_out: List[torch.Tensor] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, (lstm_h_out, lstm_c_out) = lstm(lstm_out, None if state is None else [state[0][layer_idx], state[1][layer_idx]])
            lstm_out = self.dropout(lstm_out)
            h_out.append(lstm_h_out)
            c_out.append(lstm_c_out)

        out = self.linear(lstm_out.transpose(0, 1).contiguous())
        return self.output_layer_norm(out), lengths, (torch.stack(h_out, dim = 0), torch.stack(c_out, dim = 0))

class _Joiner(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "relu") -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias = True)
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")
    
    def forward(self,
                source_encodings: torch.Tensor,
                source_lengths: torch.Tensor,
                target_encodings: torch.Tensor,
                target_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths

class RNNT(nn.Module):
    def __init__(self,
                 transcriber: Callable,
                 predictor: Callable,
                 joiner: Callable,
                 **kwargs) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner

    def forward(self,
                src: torch.Tensor,
                src_lengths: torch.Tensor,
                tgt: torch.Tensor,
                tgt_lengths: torch.Tensor,
                predictor_state: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]])
                torch.Tensor
                    joint network output, with shape
                    `(B, max output src length, max out tgt length, out dim)`
        """
        src_encodings, src_lengths = self.transcriber(
            input = src,
            lengths = src_lengths
        )
        tgt_encodings, tgt_lengths, predictor_state = self.predictor(
            input = tgt,
            lengths = tgt_lengths,
            state = predictor_state
        )
        output, src_lengths, tgt_lengths = self.joiner(
            source_encodings = src_encodings,
            source_lengths = src_lengths,
            target_encodings = tgt_encodings,
            target_lengths = tgt_lengths
        )

        return (output,
                src_lengths,
                tgt_lengths,
                predictor_state)
