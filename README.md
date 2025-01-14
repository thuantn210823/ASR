# ASR
This repository seeks to address the Automatic Speech Recognition (ASR) problem in an End-to-End (E2E) manner. The implemented E2E approaches include **Connectionist Temporal Classification (CTC)**, **Attention-based Encoder-Decoder (AED)**, and **RNN Transducer (RNN-T)**, with the latter two offering greater efficiency. Since E2E models share similar Encoder architectures, this repo utilizes the popular and highly effective **Conformer** Encoder from the '*Conformer: Convolution-augmented Transformer for Speech Recognition*' paper, paired with the AED and RNN-T Decoders. Additionally, widely used Tokenizers such as **BPE (Byte Pair Encoding)** and **WordPiece** are employed to enhance model performance.

Further tools, such as **Beam Search** and **Language Models (LM)**, which can significantly improve model quality, will be incorporated and updated in future releases.

## Installation
Clone my repo
```bash
$ git clone https://github.com/thuantn210823/ASR.git
```
Install all required libraries in the `requirements.txt` file.
```bash
cd ASR
pip install -r requirements.txt
```
## Tokenizer
I used [BPE (Byte Pair Encoding)](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) and [WordPiece](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt) tokenization methods, following the HuggingFace tutorial, and trained on the transcription of the 960h `LibriSpeech` training dataset.

For more details, please refer to the `Tokenizer.py` file in `ASR_helper` directory.
## Run
For training
```sh
cd ASR
py train.py --config_yaml YAML_PATH
```
For inference
```sh
cd ASR
py infer.py --config_yaml YAML_PATH --audio_path AUDIO_PATH
```
`Note:` If the above command doesn’t work, try replacing `py` with `python`, or the full `python.exe` path (i.e. `~/Python3xx/python.exe`) if the above code doesn't work.
## Example
```sh
cd ASR
py train.py --config_yaml conf/ConformerAED/train.yaml
```
```sh
cd ASR
py infer.py --config_yaml conf/ConformerAED/infer.yaml --audio_path example/1089-134686-0008.flac
```
`Note:` Some arguments in these `train.yaml` files are still left blank waiting for you to complete. 

Here is what you should get for the inference run above: 
```
Transcribed: the chaos in which his ardour extinguished itself was a cold indifferent knowledge of himself
```

## Pretrained Models
Pretrained models and pretrained tokenizers are offerred here, which you can find in the `pretrained` directory. 

## Results
All models were trained on the benchmark [LibriSpeech](https://www.openslr.org/12) 960h dataset. Due to the use of smaller tokenizers, architectural biases in the model, the absence of `Beam Search`, and the lack of an additional language model (`LM`), the results may be suboptimal. Below are the results of two pretrained models evaluated on the test-clean set, both utilizing `Greedy Search` for decoding.
|Model|Tokenizer|Epochs|#Params|#WER|
|:-:|:-:|:-:|:-:|:-:|
|Conformer-RNNT|BPE|60|8.79M|20.94%|
|Conformer-AED|BPE|100|10.68M|8.73%|
## Citation
Cite their great papers!
```
@article{gulati2020conformer,
  title={Conformer: Convolution-augmented transformer for speech recognition},
  author={Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and others},
  journal={arXiv preprint arXiv:2005.08100},
  year={2020}
}
```
```
@article{li2022recent,
  title={Recent advances in end-to-end automatic speech recognition},
  author={Li, Jinyu and others},
  journal={APSIPA Transactions on Signal and Information Processing},
  volume={11},
  number={1},
  year={2022},
  publisher={Now Publishers, Inc.}
}
```
```
@article{graves2012sequence,
  title={Sequence transduction with recurrent neural networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1211.3711},
  year={2012}
}
```
