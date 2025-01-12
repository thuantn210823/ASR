import yaml
import argparse
import librosa

import os

from ASR_helper.utils import transcribe
from ASR_helper.transforms import transform
from ASR_helper.Tokenizer import BPETokenizer, WPTokenizer
from ConformerRNNT import *
from ConformerAED import *

def infer(args):
    if args.tokenizer == 'BPE':
        tokenizer = BPETokenizer(args.vocab_path,
                                 args.splits_path,
                                 args.merges_path)
    elif args.tokenizer == 'WP':
        tokenizer == WPTokenizer(args.vocab_path,
                                 args.splits_path)
    else:
        "The tokenizer doesn't exist!"
        
    if args.model == 'ConformerRNNT':
        model = ConformerTransducer(input_dim = args.input_dim,
                                    output_dim = args.output_dim,
                                    time_reduction_stride = args.time_reduction_stride,
                                    conformer_input_dim = args.conformer_input_dim,
                                    conformer_ffn_dim = args.conformer_ffn_dim,
                                    conformer_num_layers = args.conformer_num_layers,
                                    conformer_num_heads = args.conformer_num_heads,
                                    conformer_depthwise_conv_kernel_size = args.conformer_kernel_size,
                                    conformer_dropout = 0.1,
                                    num_symbols = len(tokenizer.vocab),
                                    symbol_embedding_dim = args.symbol_embedding_dim,
                                    num_lstm_layers = args.num_lstm_layers,
                                    lstm_hidden_dim = args.lstm_hidden_dim,
                                    lstm_layer_norm = args.lstm_layer_norm,
                                    lstm_layer_norm_epsilon = 1e-5,
                                    lstm_dropout = 0.3,
                                    joiner_activation = args.joiner_activation,)
    elif args.model == 'ConformerAED':
        model = ConformerAED(input_dim = args.input_dim,
                             time_reduction_stride = args.time_reduction_stride,
                             conformer_input_dim = args.conformer_input_dim,
                             conformer_output_dim = args.conformer_output_dim,
                             conformer_ffn_dim = args.conformer_ffn_dim,
                             conformer_num_layers = args.conformer_num_layers,
                             conformer_num_heads = args.conformer_num_heads,
                             conformer_depthwise_conv_kernel_size = args.conformer_kernel_size,
                             conformer_dropout = 0.1,
                             vocab_size = len(tokenizer.vocab),
                             decoder_input_dim = args.decoder_input_dim,
                             decoder_ffn_dim = args.decoder_ffn_dim,
                             decoder_num_layers = args.decoder_num_layers,
                             decoder_num_heads = args.decoder_num_heads,
                             decoder_dropout = 0.1)
    else:
        raise "The model doesn't exist!!!"
    ckpt = torch.load(args.ckpt_path, map_location = 'cpu', weights_only = False)
    model.load_state_dict(ckpt['state_dict'])

    fname = os.path.split(args.audio_path)[1].split('.')[0]
    array, sr = librosa.load(args.audio_path, sr = args.sample_rate)
    array = torch.from_numpy(array.copy()).unsqueeze(0)
    trans, _ = transform(sample_rate = args.sample_rate,
                         n_fft = args.n_fft,
                         win_length = args.win_length,
                         hop_length = args.hop_length,
                         n_mels = args.n_mels,
                         augmentation = False)
    spec = trans(array).transpose(1, 2).contiguous()
    text = transcribe(model, tokenizer, spec, torch.tensor([spec.shape[1]]), args.max_tgt_lengths)
    print(f"Transcribed: {text[0]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_yaml", type = str)
    parser.add_argument("--audio_path", type = str)
    args = parser.parse_args()
    args_ = yaml.load(open(args.config_yaml, 'rb'), Loader = yaml.SafeLoader)
    args_ = argparse.Namespace(**args_)
    setattr(args_, "audio_path", args.audio_path)
    infer(args_)