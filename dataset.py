import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from joblib import Memory
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
memory = Memory('.data_cache/', verbose=0)


class WikiText103(Dataset):
    def __init__(self, split, seq_len, tokenizer_name, pad_token_id, **kwargs):
        token_seqs, self.tokenizer = prepare_wikitext_dataset(split, tokenizer_name, seq_len+1, pad_token_id)
        self.token_seqs = torch.cat(token_seqs, dim=0)

    def __len__(self):
        return self.token_seqs.size(0)

    def __getitem__(self, idx):
        context = self.token_seqs[idx, :-1]
        label = self.token_seqs[idx, 1:]
        return context, label


@memory.cache
def prepare_wikitext_dataset(split, tokenizer_name, blk_size, pad_token_id):
    logging.info(f'Preprocessing WikiText {split} dataset ...')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    corpus = dataset[split]['text']

    is_title = lambda text: text[:3] == ' = ' and text[-4:] == ' = \n' and text[3].isupper()
    text_seqs = []
    text_seq = corpus[1]  # corpus[0] is an empty string

    for i, text in enumerate(corpus[2:], start=2):
        if (corpus[i-1] == corpus[i-2] == '') and is_title(text):
            text_seqs.append(text_seq)  # Store text sequence when found a new title
            text_seq = text
        else:
            text_seq += text
    else:
        text_seqs.append(text_seq)  # The last text sequence

    logging.info(f'Tokenizing WikiText dataset with {tokenizer_name} tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=os.environ['HF_TOKEN'])
    token_seqs = []

    for text_seq in tqdm(text_seqs):
        text_tokens = tokenizer(text_seq, return_tensors='pt', return_attention_mask=False).input_ids
        text_tokens = F.pad(text_tokens, [0, blk_size - text_tokens.numel() % blk_size], 'constant', pad_token_id)
        token_seq = text_tokens.reshape(-1, blk_size).int()
        token_seqs.append(token_seq)

    return token_seqs, tokenizer


if __name__ == '__main__':
    ds = WikiText103('train', seq_len=128, tokenizer_name='meta-llama/Llama-2-7b-hf', pad_token_id=2)
    print(*[ds.tokenizer.decode(ds[i][0]) for i in range(50)], sep='\n')
    print(len(ds) // 8)
