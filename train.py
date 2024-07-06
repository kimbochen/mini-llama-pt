import math
import os
from dataclasses import asdict, dataclass, field

import torch
import wandb
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import WikiText103
from model import Transformer, MODEL_CONFIGS


@dataclass
class TrainerConfig:
    n_steps: int = 300
    bsz: int = 8
    grad_acc: int = 4
    nw: int = 8

    lr: float = 3e-4
    min_lr: float = 6e-5
    warmup_ratio: float = 0.0
    warmup_steps: int = field(init=False)

    tokenizer_name: str = 'meta-llama/Llama-2-7b-hf'
    pad_token_id: int = 2  # eos_token </s>
    ckpt_name: str = 'llama-wikitext-test'

    def __post_init__(self):
        self.warmup_steps = int(self.n_steps * self.warmup_ratio)


def linear_warmup_cosine_anneal(min_lr, lr, warmup_steps, n_steps, **kwargs):
    def get_lr_factor(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            decay_ratio = (step - warmup_steps) / (n_steps - warmup_steps)
            coeff = (1.0 + math.cos(math.pi * decay_ratio)) / 2
            lr_t = min_lr + coeff * (lr - min_lr)
            return lr_t / lr
    return get_lr_factor


def train():
    torch.manual_seed(3985)

    cfg_t = TrainerConfig()
    cfg_m = MODEL_CONFIGS['70M']
    wandb.init(project='mini-llama-pt', config={**asdict(cfg_t), **asdict(cfg_m)})

    model = Transformer(**asdict(cfg_m), **asdict(cfg_t)).to('cuda')  # torch.compile()
    dataloader = DataLoader(
        WikiText103('train', **asdict(cfg_t), **asdict(cfg_m)),
        batch_size=cfg_t.bsz, num_workers=cfg_t.nw, drop_last=True
    )
    optimizer = AdamW(model.parameters(), lr=cfg_t.lr)
    scheduler = LambdaLR(optimizer, lambda t: 1.0)  # linear_warmup_cosine_anneal(**asdict(cfg_t)))

    pbar = tqdm(total=cfg_t.n_steps)
    step_idx = 0
    model.train()

    while step_idx < cfg_t.n_steps:
        for data_b in dataloader:
            try:
                context_BT, label_BT = map(lambda z: z.to('cuda'), data_b)
                logits_BTV = model(context_BT)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.long().flatten())
                loss /= cfg_t.grad_acc

                loss.backward()
                if (step_idx + 1) % cfg_t.grad_acc == 0 or step_idx == cfg_t.n_steps - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                step_idx += 1

                wandb.log({'loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]})
                pbar.set_description(f'loss={loss.item():.4f} / lr={scheduler.get_last_lr()[0]:.2e}')
                pbar.update()

                if step_idx == cfg_t.n_steps:
                    break
            except KeyboardInterrupt:
                break

    pbar.close()
    torch.save(model.state_dict(), f'llama_{cfg_t.ckpt_name}.pth')


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = '(true | false)'
    wandb.login(key=os.environ.get('WANDB_KEY', None))  # Set WANDB_MODE="disabled" to disable
    train()
