# ASR
This is my summer project about Automatic Speech Recognition. 
## Goals:
- Learn and Build some popular End-to-End ASR architectures.
- Try using some tokenizer methods for ASR models, such as: BPE, WordPiece, etc.
- (Optional) Deploy to web-sever using AWS.
## Time:
- 2 months: Jun, 2024 - Jul, 2024
## Dataset & Environments:
- Dataset: LibriSpeech clean (100h and 360h).
- GPU: GPU P100 and GPU T4 x2 of Kaggle.
- Framework: Pytorch.
## Approach:
- There are three type of E2E ASR including CTC, RNNT and AED. I chose AED and RNNT in this project.
- In both RNNT and AED we still need an Encoder. I decided to use Conformer because it's popular and achieved SOTA performance on brenchmark dataset LibriSpeech 1000h.
- In RNNT, Predictor/Decoder I used LSTM, and in AED I used the transformer decoder with Self-attention, Cross-attention layers.
- For prediction, I want to try Greedy Search at first before using Beam Search because of LLM which is a hard work, and time-consuming.
- After all, I want to use AWS web-sever service for deployment.
## Results:
At first, I wanted to spend at least 4 months for this project - the whole summer, but many things occured that led me to close this project soon. Although my time in this project is small, I still got some encouraging results.
- I have done in bulding both RNNT and AED ASR.
- For prediction, I have done using Greedy Search.
Other works are spent for the future (i.e. Beam Search, LLM, and Deployment)!

Some inferences were maken on few samples from the evaluation dataset. I saw a positive result, but there's still a large room for improvements. For inferences, I believe that using a large language model corporating with Beam Search will enhance the results a bit. 
## Limitations:
Although my model is good to an extent, I must to tell you some limitations of my models which I've been struggled in. 
- For some reasons, AED was working very bad, though I increased the size of dataset (from 100/360h to 460h). It was struggled in convergence, after few epochs it went saturating. AED also worked bad even with BPE tokenizer, Word-Level Tokenizer. My results was gotten from using Character-Level Tokenizer.
- RNNT is a little better, but it requires large RAM capacity. In addition, RNNT model was larger in size, and slower during training than AED. I had to use 2 T4 GPU for tackling the RAM issue, and apply model parallel to endure the painful in large weight. In specific, pipelining inputs method was using for speeding up. 
## Comments: 
Some people argue that it is time-wasting and impossible in order to put an effort on a task like this in the academic environment. I do not deny it. Indeed, lacking of strong GPUs, limited datasets, and hardcore data processing might be the reasons. But I think this task is worth a try, even in the difficulty situation.

I have to tell you, I have not thing but Kaggle :)) - my new friend. But the love in AI and Speech processing made me to do this. After all, I can be satisfied with what I've done!!!
## References:
Papers:
- Li, Jinyu. "Recent advances in end-to-end automatic speech recognition." APSIPA Transactions on Signal and Information Processing 11.1 (2022).
- Graves, Alex. "Sequence transduction with recurrent neural networks." arXiv preprint arXiv:1211.3711 (2012).
- Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "Speech recognition with deep recurrent neural networks." 2013 IEEE international conference on acoustics, speech and signal processing. Ieee, 2013.
- Gulati, Anmol, et al. "Conformer: Convolution-augmented transformer for speech recognition." arXiv preprint arXiv:2005.08100 (2020).
- Dong, Linhao, Shuang Xu, and Bo Xu. "Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition." 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2018.
- Dai, Zihang, et al. "Transformer-xl: Attentive language models beyond a fixed-length context." arXiv preprint arXiv:1901.02860 (2019).
Other works, and tutorials:
- <https://keras.io/examples/audio/transformer_asr/>
- <https://github.com/park-cheol/ASR-Conformer/tree/master>
- <https://pytorch.org/audio/main/_modules/torchaudio/prototype/models/rnnt.html#conformer_rnnt_model>
- <https://huggingface.co/learn/nlp-course/chapter6/5>
Data Parallel, and Model Parallel tutorials:
- <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs>
- <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel>
