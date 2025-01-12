from typing import Callable

import yaml
import argparse

import torchaudio
from torchaudio.datasets import LIBRISPEECH

from ASR_helper import S4T as S
from ASR_helper.Tokenizer import BPETokenizer, WPTokenizer
from ASR_helper.transforms import transform
from ASR_helper.utils import create_tgt_mask
from ASR_helper.optim import NoamScheduler
from ConformerAED import *
from ConformerRNNT import *

class LS(S.SDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.root = args.root
        self.batch_size = args.batch_size
        train_dataset = []
        if args.size >= 100:
            train100 = LIBRISPEECH(root = args.root,
                                   url = 'train-clean-100',
                                   download = args.download)            
            train_dataset.append(train100)
        if args.size >= 460:
            train360 = LIBRISPEECH(root = args.root,
                                   url = 'train-clean-360',
                                   download = args.download)
            train_dataset.append(train360)
        if args.size >= 960:
            train500 = LIBRISPEECH(root = args.root,
                                   url = 'train-other-500',
                                   download = args.download)
            train_dataset.append(train500)
        self.train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        self.val_dataset = LIBRISPEECH(root = args.root,
                                       url = 'dev-clean',
                                       download = args.download)
        self.test_dataset = LIBRISPEECH(args.root,
                                        url = 'test-clean',
                                        download = args.download)
        if args.tokenizer == 'BPE':
            self.tokenizer = BPETokenizer(args.vocab_path,
                                          args.splits_path,
                                          args.merges_path)
        elif args.tokenizer == 'WP':
            self.tokenizer == WPTokenizer(args.vocab_path,
                                          args.splits_path)
        else:
            raise "The tokenizer doesn't exist!!!"
        
        self.transform, self.augment = transform(sample_rate = args.sample_rate,
                                                 n_fft = args.n_fft,
                                                 win_length = args.win_length,
                                                 hop_length = args.hop_length,
                                                 n_mels = args.n_mels,
                                                 n_time_masks = args.n_time_masks,
                                                 time_mask_param = args.time_mask_param,
                                                 n_freq_masks = args.n_freq_masks,
                                                 freq_mask_param = args.freq_mask_param)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           collate_fn = lambda x: self.collate_fn(x, True),
                                           num_workers = 4,
                                           prefetch_factor = 1,
                                           pin_memory = True,
                                           drop_last = False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = lambda x: self.collate_fn(x, False),
                                           num_workers = 4,
                                           prefetch_factor = 1,
                                           pin_memory = True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size = 1,
                                           shuffle = False,
                                           collate_fn = lambda x: self.collate_fn(x, False),
                                           num_workers = 1,
                                           prefetch_factor = 1,
                                           pin_memory = True)

    def collate_fn(self, batch, train):
        src_batch, src_lengths, tgt_batch, tgt_lengths = [], [], [], []
        for src_sample, _, tgt_sample, *_ in batch:
            tgt_sample = torch.tensor(self.tokenizer(tgt_sample, train))
            src_sample = self.transform(src_sample)
            if train and self.augment:
                src_sample = self.augment(src_sample)
            src_batch.append(src_sample.squeeze(0).transpose(0, 1).contiguous())
            tgt_batch.append(tgt_sample)
            src_lengths.append(src_sample.shape[2])
            if self.args.model == 'ConformerRNNT':
                tgt_lengths.append(len(tgt_sample)-1)

        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first = True, padding_value = -1)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first = True, padding_value = self.args.pad_idx)
        src_lengths = torch.tensor(src_lengths)
        if self.args.model == 'ConformerRNNT':
            tgt_lengths = torch.tensor(tgt_lengths)
            return src_batch, tgt_batch.type(torch.int32), src_lengths.type(torch.int32), tgt_lengths.type(torch.int32)
        else:
            return src_batch, tgt_batch.type(torch.long), src_lengths.type(torch.int32)

class ConformerTransducer_training(S.SModule, ConformerTransducer):
    def __init__(self, 
                 lr: Optional[Callable] = 0.0001,
                 pad_idx: int = 0,
                 unk_idx: int = 1,
                 bos_idx: int = 2,
                 eos_idx: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def loss(self, 
             logits: torch.Tensor, 
             targets: torch.Tensor, 
             logit_lengths: torch.Tensor, 
             target_lengths: torch.Tensor,
             blank_idx: Optional[int] = 0):
        return torchaudio.functional.rnnt_loss(logits,
                                               targets,
                                               logit_lengths,
                                               target_lengths,
                                               blank = blank_idx,
                                               reduction = 'mean')
    
    def training_step(self, batch, batch_idx):
        src, tgt, src_lengths, tgt_lengths = batch
        dec_input = tgt
        dec_target = tgt[:, 1:]

        preds, src_lengths, tgt_lengths, predictor_state = self.forward(src,
                                                                        src_lengths = src_lengths,
                                                                        tgt = dec_input,
                                                                        tgt_lengths = tgt_lengths)
        ### Ease the GPU memory occupation
        del dec_input
        del src
        del tgt
        del predictor_state
    
        loss = self.loss(preds, dec_target, src_lengths, tgt_lengths, self.pad_idx)
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
        ### Ease the GPU memory occupation
        del dec_input
        del src
        del tgt
        del predictor_state
    
        loss = self.loss(preds, dec_target.contiguous(), src_lengths, tgt_lengths, self.pad_idx)
        self.log("val_loss", loss, pbar = True, train_logging = False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0,
                                     weight_decay = 1e-6,
                                     betas = (0.9, 0.98),
                                     eps = 1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

class ConformerAED_training(S.SModule, ConformerAED):
    def __init__(self, 
                 lr: Optional[Callable] = 0.0001,
                 pad_idx: int = 0,
                 unk_idx: int = 1,
                 bos_idx: int = 2,
                 eos_idx: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def loss(self,
             y_hat: torch.Tensor, 
             y: torch.Tensor, 
             ignore_index: Optional[int] = 0):
        return nn.functional.cross_entropy(y_hat, y,
                                           ignore_index = ignore_index,
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
    
        loss = self.loss(preds.reshape(-1, preds.shape[-1]), dec_target.reshape(-1), self.pad_idx)
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
        optimizer = torch.optim.Adam(self.parameters(), lr = 0,
                                 weight_decay = 1e-6,
                                 betas = (0.9, 0.98),
                                 eps = 1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure = optimizer_closure)

        try:
            lr = self.lr.calculate_lr(epoch, batch_idx)
        except:
            lr = self.lr

        for pg in optimizer.param_groups:
            pg['lr'] = lr
        self.log('lr', lr, pbar = True, train_logging = True)

def train(args):
    ### Setup data for training
    data = LS(args)

    ### Configure model
    lr = args.lr
    if lr == 'noam':
        lr = NoamScheduler(base_lr = args.base_lr,
                           d_model = args.d_model,
                           warmup_steps = args.warmup_steps,
                           step_per_epoch = len(data.train_dataloader())//args.batch_size + 1)
    if args.model == 'ConformerRNNT':
        model = ConformerTransducer_training(input_dim = args.input_dim,
                                             output_dim = args.output_dim,
                                             time_reduction_stride = args.time_reduction_stride,
                                             conformer_input_dim = args.conformer_input_dim,
                                             conformer_ffn_dim = args.conformer_ffn_dim,
                                             conformer_num_layers = args.conformer_num_layers,
                                             conformer_num_heads = args.conformer_num_heads,
                                             conformer_depthwise_conv_kernel_size = args.conformer_kernel_size,
                                             conformer_dropout = 0.1,
                                             num_symbols = len(data.tokenizer.vocab),
                                             symbol_embedding_dim = args.symbol_embedding_dim,
                                             num_lstm_layers = args.num_lstm_layers,
                                             lstm_hidden_dim = args.lstm_hidden_dim,
                                             lstm_layer_norm = args.lstm_layer_norm,
                                             lstm_layer_norm_epsilon = 1e-5,
                                             lstm_dropout = 0.3,
                                             joiner_activation = args.joiner_activation,
                                             lr = lr,
                                             pad_idx = args.pad_idx,
                                             unk_idx = args.unk_idx,
                                             bos_idx = args.bos_idx,
                                             eos_idx = args.eos_idx)
    elif args.model == 'ConformerAED':
        model = ConformerAED_training(input_dim = args.input_dim,
                                      time_reduction_stride = args.time_reduction_stride,
                                      conformer_input_dim = args.conformer_input_dim,
                                      conformer_output_dim = args.conformer_output_dim,
                                      conformer_ffn_dim = args.conformer_ffn_dim,
                                      conformer_num_layers = args.conformer_num_layers,
                                      conformer_num_heads = args.conformer_num_heads,
                                      conformer_depthwise_conv_kernel_size = args.conformer_kernel_size,
                                      conformer_dropout = 0.1,
                                      vocab_size = len(data.tokenizer.vocab),
                                      decoder_input_dim = args.decoder_input_dim,
                                      decoder_ffn_dim = args.decoder_ffn_dim,
                                      decoder_num_layers = args.decoder_num_layers,
                                      decoder_num_heads = args.decoder_num_heads,
                                      decoder_dropout = 0.1,
                                      lr = lr,
                                      pad_idx = args.pad_idx,
                                      unk_idx = args.unk_idx,
                                      bos_idx = args.bos_idx,
                                      eos_idx = args.eos_idx)
    else:
        raise "The model doesn't exist!!!"

    ### Train
    torch.manual_seed(args.seed)
    checkpoint_callback = S.ModelCheckpoint(dirpath = args.ckpt_dir,
                                            save_top_k = 10, monitor = 'val_loss',
                                            mode = 'min',
                                            filename = f'{args.model}-epoch:%02d-val_loss:%.4f')
    trainer = S.Trainer(accelerator = args.device,
                        callbacks = [checkpoint_callback],
                        enable_checkpointing = True,
                        max_epochs = args.max_epochs,
                        gradient_clip_val = args.gradient_clip_val)
    history = trainer.fit(model, data)
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_yaml", type = str)
    args = parser.parse_args()
    args = yaml.load(open(args.config_yaml, 'rb'), Loader = yaml.SafeLoader)
    args = argparse.Namespace(**args)
    train(args)