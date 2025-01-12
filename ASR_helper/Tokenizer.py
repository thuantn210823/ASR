from typing import List

import collections
import copy
import pickle

def _compute_pair_freqs(splits, word_freqs):
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

def _merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a + b] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits

def BPE(base_vocab: List[str],
        word_freqs: dict,
        vocab_size: int):
    vocab = copy.deepcopy(base_vocab)
    # splits
    splits = {word: list(word) for word in word_freqs.keys()}

    # merge and add
    merges = {}
    while len(vocab) < vocab_size:
        pair_freqs = _compute_pair_freqs(splits, word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = _merge_pair(*best_pair, splits, word_freqs)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    return vocab, splits, merges

class BPETokenizer:
    def __init__(self, 
                 vocab_pickle: str, 
                 splits_pickle: str,
                 merges_pickle: str,
                 special_token: str = '@',
                 unk_idx: int = 1,
                 bos_idx: int = 2,
                 eos_idx: int = 3):
        self.vocab = pickle.load(open(vocab_pickle, 'rb'))
        self.splits = pickle.load(open(splits_pickle, 'rb'))
        self.merges = pickle.load(open(merges_pickle, 'rb'))
        self.special_token = special_token
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
    
    def __call__(self, 
                 text: str,
                 train: bool = True):
        if train:
            text = ' ' + text
            text = text.lower().replace(' ', ' @')
            words = text.split()
            tokens = []
            for word in words:
                tokens += self.splits.get(word, list(word))
        else:
            tokens = self.tokenize(text)
        return [self.bos_idx] + [self.token_to_idx.get(token, self.unk_idx) for token in tokens] + [self.eos_idx]
    
    def stoi(self, 
             text: str,
             train: bool):
        self.__call__(text, train)
    
    def itos(self,
             indices: List[int]):
        tokens = [self.vocab[idx] for idx in indices]
        text = ''.join(tokens).replace('@', ' ')
        if text[0] == ' ':
            text = text[1:]
        return text

    def tokenize(self, 
                 text: str):
        text = ' ' + text
        text = text.lower().replace(' ', ' @')
        words = text.split()
        splits = [[l for l in word] for word in words]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i+2:]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])

def _compute_pair_scores(splits, word_freqs):
    letter_freqs = collections.defaultdict(int)
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq/(letter_freqs[pair[0]]*letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

def _WP_merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def WP(base_vocab: List[str],
       word_freqs: dict,
       vocab_size: int):
    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")
    alphabet.sort()
    vocab = base_vocab + alphabet.copy()
    splits = {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in word_freqs.keys()
    }
    while len(vocab) < vocab_size:
        scores = _compute_pair_scores(splits, word_freqs)
        best_pair, max_score = "", None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score
        splits = _WP_merge_pair(*best_pair, splits, word_freqs)
        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith('##')
            else best_pair[0] + best_pair[1]
        )
        vocab.append(new_token)
    return vocab, splits

class WPTokenizer:
    def __init__(self, 
                 vocab_pickle: str, 
                 splits_pickle: str,
                 unk_idx: int = 1,
                 bos_idx: int = 2,
                 eos_idx: int = 3):
        self.vocab = pickle.load(open(vocab_pickle, 'rb'))
        self.splits = pickle.load(open(splits_pickle, 'rb'))
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
    
    def __call__(self, 
                 text: str,
                 train: bool = True):
        if train:
            words = text.lower().split()
            tokens = []
            for word in words:
                tokens += self.splits.get(word, ['<UNK>'])
        else:
            tokens = self.tokenize(text)
        return [self.bos_idx] + [self.token_to_idx.get(token, self.unk_idx) for token in tokens] + [self.eos_idx]
    
    def stoi(self, 
             text: str,
             train: bool):
        self.__call__(text, train)
    
    def itos(self,
             indices: List[int]):
        tokens = [self.vocab[idx] for idx in indices]
        text = '@'.join(tokens)
        text = text.replace('@##', '')
        text = text.replace('@', ' ')
        return text

    def tokenize(self, 
                 text: str):
        words = text.lower().split()
        encoded_words = [self._encode_word(word) for word in words]
        return sum(encoded_words, [])
    
    def _encode_word(self,
                     word: str):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["<UNK>"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens
