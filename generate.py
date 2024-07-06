import os
import sys
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import Transformer, MODEL_CONFIGS
from train import TrainerConfig


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256


@torch.inference_mode()
def generate():
    prompt = sys.argv[2]

    cfg_g = GenerationConfig()
    cfg_t = TrainerConfig()
    cfg_m = MODEL_CONFIGS['70M']

    tokenizer = AutoTokenizer.from_pretrained(cfg_t.tokenizer_name, token=os.environ['HF_TOKEN'])
    model = Transformer(**asdict(cfg_m), **asdict(cfg_t)).to('cuda')  # torch.compile()
    model.load_state_dict(torch.load(sys.argv[1]))

    tokens_BT = tokenizer(prompt, return_tensors='pt', return_attention_mask=False).input_ids
    tokens_BT = tokens_BT.to('cuda')

    model.eval()
    new_tokens = 0
    new_token_BT = None
    while new_tokens < cfg_g.max_new_tokens and new_token_BT != tokenizer.eos_token_id:
        logits_BTC = model(tokens_BT)
        new_token_BT = logits_BTC[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens_BT = torch.cat([tokens_BT, new_token_BT], dim=1)[:, -cfg_m.seq_len:]
        new_tokens += 1

    print(tokenizer.decode(tokens_BT.squeeze(), skip_special_tokens=True))


if __name__ == '__main__':
    generate()
