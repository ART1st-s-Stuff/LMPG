import torch
from torch import nn, optim
import os
import string
import random
import re
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, Lock

from model.model import HRNN
from model.decoder import FFNNDecoder

from datasets import load_dataset, Dataset, NamedSplit


def load_checkpoint(checkpoint_path: str):
    print("Loading model.")
    if not os.path.exists(checkpoint_path):
        model = HRNN.tiny()
        decoder1 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=2048,
            num_layers=4,
            dim_output=1,
            dropout=0.3,
            type="sigmoid"
        )
        decoder2 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=2048,
            num_layers=4,
            dim_output=1,
            dropout=0.3,
            type="sigmoid"
        )
        decoder3 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=2048,
            num_layers=4,
            dim_output=32,
            dropout=0.3,
            type="sigmoid"
        )
    else:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        decoder1 = checkpoint["decoder1"]
        decoder2 = checkpoint["decoder2"]
        decoder3 = checkpoint["decoder3"]
    print(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, decoder1, decoder2, decoder3

def as_utf32_tensor(s: str):
    # To 32-bit 1-hot tensor
    tensor = torch.zeros(len(s), 32, dtype=torch.float32)
    for i, c in enumerate(s):
        utf32_c = ord(c)
        # utf32_c to binary
        tensor[i] = torch.tensor([(utf32_c >> j) & 1 for j in range(32)], dtype=torch.float32)
    return tensor

def build_vocabulary_dataset(data: Dataset):
    if os.path.exists("dataset/vocab.ds"):
        print("Loading vocabulary dataset from cache.")
        return torch.load("dataset/vocab.ds")
    vocab = set()
    ds = []
    for item in tqdm(data, desc="Building vocabulary dataset stage 1"):
        text : str = item["text"]
        words = re.split(f"[{re.escape(string.punctuation)} \n\t]", text)
        vocab.update(words)
    for v in tqdm(vocab, desc="Building vocabulary dataset stage 2"):
        if len(v) == 0:
            continue
        y = torch.ones(len(v), dtype=torch.float32)
        y[-1] = 0  # End token
        ds.append((0, as_utf32_tensor(v), y))
    torch.save(ds, "dataset/vocab.ds")
    return ds

def sanity(text: str):
    remove_period = text.rstrip('.')
    if len(remove_period) == 0:
        return None, None
    y = torch.ones(len(remove_period), dtype=torch.float32)
    y[-1] = 0  # End token
    return as_utf32_tensor(remove_period), y

def cloze(text: str):
    def randomly_remove_word(s: str):
        words = re.split(f"[{re.escape(string.punctuation)} \n\t]", s)
        if len(words) <= 1:
            return None, None
        idx = random.randint(0, len(words) - 1)
        answer = words[idx]
        words[idx] = "<BLANK>"
        return " ".join(words), answer
    
    sentence, answer = randomly_remove_word(text)
    if sentence is None or answer is None:
        return None, None
    sentence = "<CLOZE TASK>: " + sentence
    answer_tensor = as_utf32_tensor(answer)
    answer_tensor = torch.cat([answer_tensor, torch.zeros(32, dtype=torch.float32)], dim=0)  # End token
    return as_utf32_tensor(sentence), answer_tensor

def train(model: HRNN, decoder1: FFNNDecoder, decoder2: FFNNDecoder, decoder3: FFNNDecoder):
    ds = load_dataset("agentlans/high-quality-english-sentences")
    ds = ds["train"]
    
    vocabulary_ds = build_vocabulary_dataset(ds)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(decoder1.parameters()) + list(decoder2.parameters()) + list(decoder3.parameters()), lr=0.001)
    
    
    model = model.to(device)
    decoder1 = decoder1.to(device)
    decoder2 = decoder2.to(device)
    decoder3 = decoder3.to(device)
    
    model.train()
    decoder1.train()
    decoder2.train()
    decoder3.train()
    
    model.zero_grad()
    decoder1.zero_grad()
    decoder2.zero_grad()
    decoder3.zero_grad()
    
    print("Building datasets...")

    #random.shuffle(train)
    losses = [-1, -1, -1]
    i = 0

    print("Start training process...")
    
    training_ds = list(ds) + vocabulary_ds
    random.shuffle(training_ds)

    for v in tqdm(training_ds):
        if isinstance(v, tuple):
            _, x, y = v
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x, post_processing=decoder1).squeeze(0)
            task = 0
        else:
            randval = random.random()
            if randval < 0.9:
                x, y = sanity(v["text"])
                if x is None or y is None:
                    continue
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x, post_processing=decoder2).squeeze(0)
                task = 1
            else:
                x, y = cloze(v["text"])
                if x is None or y is None:
                    continue
                x = x.to(device)
                y = y.to(device)
                y_hat = model.self_regression(x, max_output=y.shape[0], halt_if=lambda _: False, post_processing=decoder3)
                task = 2

        # Compute loss
        #print(task, y_hat.shape, y.shape)
        #print(y_hat, y)
        loss = loss_fn(y_hat, y)
        losses[task] = loss.item() if losses[task] < 0 else 0.7 * losses[task] + 0.3 * loss.item()

        # Backward pass
        loss.backward()

        optimizer.step()
        
        # Display progress bar and data per 1000 steps
        if i % 1000 == 0:
            print(f"Step {i}: Losses: Vocabulary={losses[0]:.4f}, Sentence Sanity={losses[1]:.4f}, Cloze={losses[2]:.4f}")
            torch.save({
                "model": model,
                "decoder1": decoder1,
                "decoder2": decoder2,
                "decoder3": decoder3,
            }, "checkpoints/word_identification.pth")
        i += 1
        
        
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
if __name__ == "__main__":
    set_seed(114514)
    ds = load_dataset("agentlans/high-quality-english-sentences")
    packed_models = load_checkpoint("checkpoints/word_identification.pth")
    train(*packed_models, ds["train"])