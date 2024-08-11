from typing import Optional, Callable, List, Tuple

import torch
from torch import nn
import torchaudio
import torchmetrics

import os

import torchmetrics.text

sys.path.append('./asr_helper')
from Conformer import ConformerEncoderLayer, ConformerEncoder
from RNNT import _TimeReduction, _Predictor, _Joiner, RNNT
from Tokenizer import BPETokenizer
import S4T as S

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

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

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, root, subset = 'train'):
        super().__init__()
        self.subset = subset
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url = subset)

    def __getitem__(self, idx):
        wav, _, text, *_ = self.dataset[idx]
        wav = TRANSFORM(wav)
        if 'train' in self.subset:
            wav = TRAIN_TRANSFORM(wav)
        return wav, text

    def __len__(self):
        return len(self.dataset)

class LS460(S.SDataModule):
    def __init__(self, root, batch_size):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        train100 = LibriSpeech(root,
                                 subset = 'train-clean-100')
        train360 = LibriSpeech(root,
                                 subset = 'train-clean-360')
        self.train_dataset = torch.utils.data.ConcatDataset([train100, train360])
        self.val_dataset = LibriSpeech(root,
                               subset = 'dev-clean')
        self.test_dataset = LibriSpeech(root,
                                subset = 'test-clean')
        self.tokenizer = BPETokenizer('./asr_helper/BPEvocab512.pkl',
                                      './asr_helper/BPEsplits512.pkl',
                                      './asr_helper/BPEmerges512.pkl')

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
        src_batch, src_lengths, tgt_batch, tgt_lengths = [], [], [], []
        for src_sample, tgt_sample in batch:
            tgt_sample = torch.tensor(self.tokenizer(tgt_sample))
            src_batch.append(src_sample.squeeze(0).transpose(0, 1).contiguous())
            tgt_batch.append(tgt_sample)
            src_lengths.append(src_sample.shape[2])
            tgt_lengths.append(len(tgt_sample)-1)

        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first = True, padding_value = 0)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first = True, padding_value = PAD_IDX)
        src_lengths = torch.tensor(src_lengths)
        tgt_lengths = torch.tensor(tgt_lengths)
        return src_batch, tgt_batch.type(torch.int32), src_lengths.type(torch.int32), tgt_lengths.type(torch.int32)

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
                 joiner_activation: str = 'relu') -> None:
        transcriber = _ConformerEncoder(input_dim,
                                             output_dim,
                                             time_reduction_stride,
                                             conformer_input_dim,
                                             conformer_ffn_dim,
                                             conformer_num_layers,
                                             conformer_num_heads,
                                             conformer_depthwise_conv_kernel_size,
                                             conformer_dropout).to('cuda:0')
        predictor = _Predictor(num_symbols,
                                    output_dim,
                                    symbol_embedding_dim,
                                    num_lstm_layers,
                                    lstm_hidden_dim,
                                    lstm_layer_norm,
                                    lstm_layer_norm_epsilon,
                                    lstm_dropout).to('cuda:0')
        joiner = _Joiner(output_dim, num_symbols, activation = joiner_activation).to('cuda:1')
        super().__init__(transcriber,
                         predictor,
                         joiner)
    
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
        ## Model parallel method to extend the model size
        src_encodings, src_lengths = self.transcriber(
            input = src.to('cuda:0'),
            lengths = src_lengths.to('cuda:0')
        )
        tgt_encodings, tgt_lengths, predictor_state = self.predictor(
            input = tgt.to('cuda:0'),
            lengths = tgt_lengths.to('cuda:0'),
            state = [x.to('cuda:0') for x in predictor_state] \
                           if predictor_state is not None else None
        )
        output, src_lengths, tgt_lengths = self.joiner(
            source_encodings = src_encodings.to('cuda:1'),
            source_lengths = src_lengths.to('cuda:1'),
            target_encodings = tgt_encodings.to('cuda:1'),
            target_lengths = tgt_lengths.to('cuda:1')
        )

        return (output,
                src_lengths,
                tgt_lengths,
                predictor_state)

    def generate(self, 
                 input: torch.Tensor, 
                 input_lengths: torch.Tensor,
                 max_tgt_lengths: torch.Tensor = 400):
        y_batch = []
        B = len(input)
        enc_out, enc_lengths = self.transcriber(input, input_lengths)
        for b in range(B):
            t = 0; u = 0;
            y = [BOS_IDX]
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
                if argmax == PAD_IDX:
                    t += 1
                elif argmax == EOS_IDX:
                    break
                else:
                    u += 1
                    y.append(argmax)
            y_batch.append(y[1:])
        return y_batch

# Model for training
class ConformerTransducer_training(S.SModule, ConformerTransducer):
    def __init__(self, 
                 split_size: int,
                 lr: Optional[Callable] = 0.0001,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.split_size = split_size

    def forward(self,
                src: torch.Tensor,
                src_lengths: torch.Tensor,
                tgt: torch.Tensor,
                tgt_lengths: torch.Tensor,
                predictor_state: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        ## pipelining inputs
        src_splits = iter(src.split(self.split_size, dim = 0))
        src_len_splits = iter(src_lengths.split(self.split_size, dim = 0))
        tgt_splits = iter(tgt.split(self.split_size, dim = 0))
        tgt_len_splits = iter(tgt_lengths.split(self.split_size, dim = 0))
        # predictor_state_splits = iter()

        src_next = next(src_splits)
        src_len_next = next(src_len_splits)
        tgt_next = next(tgt_splits)
        tgt_len_next = next(tgt_len_splits)
        src_prev, src_len_prev = self.transcriber(input = src_next.to('cuda:0'),
                                                  lengths = src_len_next.to('cuda:0'))
        tgt_prev, tgt_len_prev, predictor_state_prev = self.predictor(input = tgt_next.to('cuda:0'),
                                                                      lengths = tgt_len_next.to('cuda:0'),
                                                                      state = predictor_state)
        ret_output = []
        ret_src_lens = []
        ret_tgt_lens = []
        for (src_next, src_len_next, tgt_next, tgt_len_next) in zip(src_splits, 
                                                                    src_len_splits,
                                                                    tgt_splits,
                                                                    tgt_len_splits):
            out, src_lens, tgt_lens = self.joiner(src_prev.to('cuda:1'),
                                                  src_len_prev.to('cuda:1'),
                                                  tgt_prev.to('cuda:1'),
                                                  tgt_len_prev.to('cuda:1'))
            ret_output.append(out)
            ret_src_lens.append(src_lens)
            ret_tgt_lens.append(tgt_lens)
            
            src_prev, src_len_prev = self.transcriber(input = src_next.to('cuda:0'),
                                                  lengths = src_len_next.to('cuda:0'))
            tgt_prev, tgt_len_prev, predictor_state_prev = self.predictor(input = tgt_next.to('cuda:0'),
                                                                      lengths = tgt_len_next.to('cuda:0'),
                                                                      state = predictor_state)
        out, src_lens, tgt_lens = self.joiner(src_prev.to('cuda:1'),
                                              src_len_prev.to('cuda:1'),
                                              tgt_prev.to('cuda:1'),
                                              tgt_len_prev.to('cuda:1'))
        ret_output.append(out)
        ret_src_lens.append(src_lens)
        ret_tgt_lens.append(tgt_lens)
        return torch.cat(ret_output), torch.cat(ret_src_lens), torch.cat(ret_tgt_lens), predictor_state

    def loss(self, logits, targets, logit_lengths, target_lengths):
        return torchaudio.functional.rnnt_loss(logits,
                                               targets,
                                               logit_lengths,
                                               target_lengths,
                                               blank = PAD_IDX,
                                               reduction = 'mean')

    def training_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        dec_input = tgt
        dec_target = tgt[:, 1:]

        preds, src_lengths, tgt_lengths, predictor_state = self.forward(src,
                                                                        src_lengths = src_lengths,
                                                                        tgt = dec_input,
                                                                        tgt_lengths = tgt_lengths)
    
        loss = self.loss(preds, dec_target.contiguous().to('cuda:1'), src_lengths, tgt_lengths)
        self.log("train_loss", loss, pbar = True, train_logging = True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        dec_input = tgt
        dec_target = tgt[:, 1:]

        preds, src_lengths, tgt_lengths, predictor_state = self.forward(src,
                                                                        src_lengths = src_lengths,
                                                                        tgt = dec_input,
                                                                        tgt_lengths = tgt_lengths)
    
        loss = self.loss(preds, dec_target.contiguous().to('cuda:1'), src_lengths, tgt_lengths)
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

if __name__ == '__main__':
    ## data
    data = data = LS460('/kaggle/input/librispeech-clean', 8)

    checkpoint_callback = S.ModelCheckpoint(dirpath = './',
                                            save_top_k = 7, monitor = 'val_loss',
                                            mode = 'min',
                                            filename = 'conformer_rnnt-10m-bpe-ls100-epoch:%02d-val_loss:%.4f')
    lr = CustomLRScheduler()
    model = ConformerTransducer_training(input_dim = 80,
                                         output_dim = 512,
                                         time_reduction_stride = 4,
                                         conformer_input_dim = 256,
                                         conformer_ffn_dim = 512,
                                         conformer_num_layers = 8,
                                         conformer_num_heads = 4,
                                         conformer_depthwise_conv_kernel_size = 31,
                                         conformer_dropout = 0.1,
                                         num_symbols = len(data.tokenizer.vocab),
                                         symbol_embedding_dim = 128,
                                         num_lstm_layers = 1,
                                         lstm_hidden_dim = 320,
                                         lstm_layer_norm = True,
                                         lstm_layer_norm_epsilon = 1e-5,
                                         lstm_dropout = 0.3,
                                         joiner_activation = "tanh",
                                         lr = lr,
                                         split_size = 8)
    ## Print out the model size
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    ## training
    trainer = S.Trainer(callbacks = [checkpoint_callback],
                        enable_checkpointing = True,
                        max_epochs = 100,
                        gradient_clip_val = 1,
                        accelerator = "gpu",
                        devices = [0, 1])
    
    trainer.fit(model, data)

    ## Making simple inference using greedy search
    test_dataset = data.test_dataset
    spec0, text0 = test_dataset[0]
    spec0 = spec0.transpose(1, 2).contiguous()
    pred0 = transcribe(model, data.tokenizer, spec0, torch.tensor([spec0.shape[1]]), 400)[0]
    print("Ground truth", text0)
    print("Predicted", pred0)
    print("Wer:", torchmetrics.text.WordErrorRate()(pred0, text0))