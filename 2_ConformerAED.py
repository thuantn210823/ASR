from typing import Optional, Callable, List, Tuple

import torch
from torch import nn
import torchaudio
import torchmetrics

import math

import os
import sys

from Conformer import ConformerEncoderLayer, ConformerEncoder
from Transformer import CustomTransformerDecoderLayer, CustomTransformerDecoder
from RNNT import _TimeReduction
from Embedding import PositionalEncoding
import S4T as S

# Data augmentation
TRANSFORM = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                                 n_fft = 512,
                                                                 win_length = 400,
                                                                 hop_length = 160,
                                                                 n_mels = 80),
                          torchaudio.transforms.AmplitudeToDB())
TRAIN_TRANSFORM = torchaudio.transforms.SpecAugment(n_time_masks = 10,
                                      time_mask_param = 10,
                                      n_freq_masks = 1,
                                      freq_mask_param = 27)

# special token indexes
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

class VectorizeChar:
    def __init__(self):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", "'"]
        )
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = "<" + text + ">"
        encoded = []
        return [self.char_to_idx.get(ch, 1) for ch in text]

    def get_vocabulary(self):
        return self.vocab
    
    def itos(self, indices):
        chars = []
        for idx in indices:
            chars.append(self.vocab[idx])
        text = "".join(chars).replace("<", "").replace(">","")
        return text

class LibriSpeech(torch.utils.data.Dataset):
    """
    LibriSpeech Dataset.

    Parameters:
    root: str - Path to the directory where the dataset is found or downloaded.
    subset: str - The type of the dataset (e.g. 'train-clean-100').
    """
    def __init__(self, 
                 root: str, 
                 subset: str = 'train'):
        super().__init__()
        self.subset = subset
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url = subset)

    def __getitem__(self, idx: int):
        """
        Parameters:
        idx: int - The index of the sample to be loaded
        Returns:
        Tuple of (Tensor, string)
        """
        wav, _, text, *_ = self.dataset[idx]
        wav = TRANSFORM(wav)
        if 'train' in self.subset:
            wav = TRAIN_TRANSFORM(wav)
        return wav, text

class LS460(S.SDataModule):
    """
    Dataloader of Librispeech dataset 460h.
    
    Parameters:
    root: str - Path to the directory where the dataset is found or downloaded.
    batch_size: int - The number of samples per mini-batch.
    """
    def __init__(self, root, batch_size):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        train100 = LibriSpeech(root,
                                 subset = 'train-clean-100')
        train360 = LibriSpeech(root,
                                 subset = 'train-clean-360')
        self.train_dataset = torch.utils.data.ConcatDataset([train100, train360])
        # self.train_dataset = train100
        self.val_dataset = LibriSpeech(root,
                               subset = 'dev-clean')
        self.test_dataset = LibriSpeech(root,
                                subset = 'test-clean')
        self.tokenizer = VectorizeChar()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           collate_fn = self.collate_fn,
                                           num_workers = 4,
                                           prefetch_factor = 1,
                                           pin_memory = True,
                                           drop_last = False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = self.collate_fn,
                                           num_workers = 4,
                                           prefetch_factor = 1,
                                           pin_memory = True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size = 1,
                                           shuffle = False,
                                           collate_fn = self.collate_fn,
                                           num_workers = 1,
                                           prefetch_factor = 1,
                                           pin_memory = True)

    def collate_fn(self, batch):
        """
        Processing the list of samples to batch.
        Returns:
        Tuple of the following items:
            Tensor - Batch of source spectrograms.
            Tensor - Batch of encoded labels.
            Tensor - The lengths of source spectrograms in batch.
        """
        src_batch, src_lengths, tgt_batch = [], [], []
        for src_sample, tgt_sample in batch:
            tgt_sample = torch.tensor(self.tokenizer(tgt_sample.lower()))
            src_batch.append(src_sample.squeeze(0).transpose(0, 1).contiguous())
            tgt_batch.append(tgt_sample)
            src_lengths.append(src_sample.shape[2])

        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first = True, padding_value = 0)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first = True, padding_value = PAD_IDX)
        src_lengths = torch.tensor(src_lengths)
        return src_batch, tgt_batch.type(torch.long), src_lengths.type(torch.long)

class _ConformerEncoder(nn.Module):
    """
    Encoder or Transcriber of model used Conformer to extract informations.

    Parameters:
    input_dim: int - The input dimension.
    output_dim: int - The output dimention.
    time_reduction_stride: int - The number of frames will be concatenated.
    conformer_input_dim: int - The embedding dimension.
    conformer_ffn_dim: int - The dimention of the feedforward network.
    conformer_num_layers: int - The number of sub-encoder-layers.
    conformer_num_heads: int - The number of heads in the multiheadattention models.
    conformer_depthwise_conv_kernel_size: int - The kernel size of convolution layers.
    conformer_dropout: float - The dropout value
    """
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
        Paramameters:
        input: Tensor - Audio features, shape (N, T, D).
        length: Tensor - Correspoding Lengths of the audio features, shape (N,)
        """
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, _, key_padding_mask = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, key_padding_mask

class TokenEmbedding(nn.Module):
    """
    Token embedding module.

    Parameters:
    vocab_size: int - The size of vocabulary.
    emb_size: int - The embedding size.
    """
    def __init__(self, 
                 vocab_size: int, 
                 emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long())*math.sqrt(self.emb_size)

def generate_square_subsequent_mask(sz):
    """
    Making causal mask for cross-attention layers.

    Parameters:
    sz: int: The number of tokens in the taget sequence.
    """
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False).type(torch.bool)
    return mask

def create_tgt_mask(tgt):
    """
    Create mask for attention layers, including padding mask, and causal mask.

    Parameters:
    tgt: Tensor - The targets, shape (N, T, D) 
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

    tgt_padding_mask = (tgt == PAD_IDX)
    return tgt_mask.to(tgt.device), tgt_padding_mask.to(tgt.device)

class _TransformerDecoder(nn.Module):
    """
    Transformer Decoder

    Parameters:
    vocab_size: int - The size of the vocabulary.
    num_hiddens: int - The number of expected features in the target.
    ffn_num_hiddens: int - The dimension of the feedforward network.
    num_heads: int - The number of heads in the multiheadattention models.
    num_blks: int - The number of the transformer decoder layer.
    dropout: float - The dropout value.
    """
    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_blks: int,
                 dropout: float = 0.1):
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
    """
    Attention-based Encoder-Decoder ASR using Conformer as Encoder.
    
    Parameters:
    input_dim: int - The input dimension.
    output_dim: int - The output dimention.
    time_reduction_stride: int - The number of frames will be concatenated.
    conformer_input_dim: int - The embedding dimension.
    conformer_ffn_dim: int - The dimention of the feedforward network.
    conformer_num_layers: int - The number of sub-encoder-layers.
    conformer_num_heads: int - The number of heads in the multiheadattention models.
    conformer_depthwise_conv_kernel_size: int - The kernel size of convolution layers.
    conformer_dropout: float - The dropout value.
    vocab_size: int - The size of the vocabulary.
    decoder_input_dim: int - The number of expected features in the target.
    decoder_ffn_dim: int - The dimension of the feedforward network.
    decoder_num_heads: int - The number of heads in the multiheadattention models.
    decoder_num_layers: int - The number of the transformer decoder layer.
    decoder_dropout: float - The dropout value.
    """
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
                 decoder_dropout: float) -> None:
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
                                         conformer_dropout)
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
                 start_symbol: int = 0):
        """
        Generating output given audio feature.
        """
        memory, memory_key_padding_mask = self.encoder(src, src_lengths)
        if self.conformer_output_dim != self.decoder_input_dim:
            memory = self.cross_linear(memory)
        dec_input = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(src)
        dec_logits = []
        for i in range(max_tgt_lengths-1):
            tgt_mask = generate_square_subsequent_mask(dec_input.shape[1])
            out = self.decoder(dec_input, memory, tgt_mask, memory_key_padding_mask = memory_key_padding_mask)
            out = out.transpose(1, 2).contiguous()
            prob = self.classifier(out[:, :, -1])
            _, next_word = torch.max(prob, dim = -1)
            dec_logits.append(next_word.item())
        
            dec_input = torch.cat([dec_input, torch.ones(1, 1).type_as(src.data).fill_(next_word.item())], dim = 1)
            if next_word == EOS_IDX:
                break
        return dec_logits

## Training model
class ConformerAED_training(S.SModule, ConformerAED):
    def __init__(self,
                 lr: Optional[Callable] = 0.0001,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr

    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y,
                                           ignore_index = PAD_IDX,
                                           reduction = 'mean',
                                           label_smoothing = 0.1)

    def training_step(self, batch, batch_idx):
        src, tgt, src_lengths = batch
        dec_input = tgt[:, :-1]
        dec_target = tgt[:, 1:]

        tgt_mask, tgt_padding_mask = create_tgt_mask(dec_input)
        preds = self.forward(src, dec_input,
                             src_lengths = src_lengths,
                             tgt_mask = tgt_mask,
                             tgt_padding_mask = tgt_padding_mask)
    
        loss = self.loss(preds.reshape(-1, preds.shape[-1]), dec_target.reshape(-1))
        self.log("train_loss", loss, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, src_lengths = batch
        dec_input = tgt[:, :-1]
        dec_target = tgt[:, 1:]

        tgt_mask, tgt_padding_mask = create_tgt_mask(dec_input)
        preds = self.forward(src, dec_input,
                             src_lengths = src_lengths,
                             tgt_mask = tgt_mask,
                             tgt_padding_mask = tgt_padding_mask)

        loss = self.loss(preds.reshape(-1, preds.shape[-1]), dec_target.reshape(-1))
        self.log("val_loss", loss, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0005,
                                     weight_decay = 1e-6,
                                     betas = (0.9, 0.98),
                                     eps = 1e-9)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        # manually warm up lr without a scheduler
        lr = self.lr.calculate_lr(epoch)

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

    def apply_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class CustomLRScheduler:
    def __init__(self,
                 init_lr = 0.0005,
                 lr_after_warmup = 0.001,
                 final_lr = 0.0001,
                 warmup_epochs = 5,
                 decay_epochs = 100):
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs

    def calculate_lr(self, epoch):
        """
        Linear warm up - linear decay
        """
        warmup_lr = self.init_lr + ((self.lr_after_warmup - self.init_lr)/(self.warmup_epochs - 1))*epoch
        decay_lr = max(self.final_lr,
                       self.lr_after_warmup
                       - (epoch - self.warmup_epochs)
                       *(self.lr_after_warmup - self.final_lr)
                       /self.decay_epochs)
        return min(warmup_lr, decay_lr)

# transcribe
def transcribe(model: Callable,
               tokenizer: Callable,
               src: torch.Tensor,
               src_lengths: Optional[torch.Tensor] = None,
               max_tgt_lengths: int = 400):
    """
    Transcribing audio to text.

    Parameters:
    model: Callable - Transformer-Transducer Model.
    tokenizer: Callable - Tokenizer.
    src: Tensor - Input audio features.
    src_lengths: Optional[Tensor] - The correspoding lengths of the audio features.
    max_tgt_lengths: int - The maximum target lengths for prediction.
    """
    model.eval()
    tgt_tokens = model.generate(src, src_lengths, max_tgt_lengths, BOS_IDX)
    return tokenizer.itos(tgt_tokens)

if __name__ == '__main__':
    ## data
    data = LS460('/kaggle/input/librispeech-clean', 16)

    checkpoint_callback = S.ModelCheckpoint(dirpath = './',
                                      save_top_k = 7, monitor = 'val_loss',
                                      mode = 'min',
                                      filename = 'conformer_aed-char-ls100-epoch:%02d-val_loss:%.4f')
    lr = CustomLRScheduler()
    model = ConformerAED_training(input_dim = 80,
                                  time_reduction_stride = 4,
                                  conformer_input_dim = 512,
                                  conformer_output_dim = 512,
                                  conformer_ffn_dim = 512,
                                  conformer_num_layers = 4,
                                  conformer_num_heads = 4,
                                  conformer_depthwise_conv_kernel_size = 31,
                                  conformer_dropout = 0.1,
                                  vocab_size = len(data.tokenizer.vocab),
                                  decoder_input_dim = 512,
                                  decoder_ffn_dim = 512,
                                  decoder_num_layers = 4,
                                  decoder_num_heads = 4,
                                  decoder_dropout = 0.1,
                                  lr = lr)
    model.apply_init()

    ## Print out the model size 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    ## training
    trainer = S.Trainer(callbacks = [checkpoint_callback],
                   enable_checkpointing = True,
                   max_epochs = 100,
                   gradient_clip_val = 1,
                   accelerator = 'gpu')
    trainer.fit(model, data)

    ## Making simple inference using greedy search
    test_dataset = data.test_dataset
    spec0, text0 = test_dataset[0]
    spec0 = spec0.transpose(1, 2).contiguous()
    pred0 = transcribe(model, data.tokenizer, spec0, torch.tensor([spec0.shape[1]]), 400)
    print("Ground truth", text0)
    print("Predicted", pred0)
    print("Wer:", torchmetrics.text.WordErrorRate()(pred0, text0))