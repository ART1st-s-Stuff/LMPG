import torch
from torch import nn, optim
import os
import string
import random
import re
from tqdm import tqdm

from model.model import HRNN
from model.decoder import FFNNDecoder

from datasets import load_dataset

ds = load_dataset("agentlans/high-quality-english-sentences")

def load_checkpoint(checkpoint_path: str):
    print("Loading model.")
    if not os.path.exists(checkpoint_path):
        model = HRNN.default()
        decoder1 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=256,
            num_layers=4,
            dim_output=1,
            dropout=0.3,
            type="sigmoid"
        )
        decoder2 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=256,
            num_layers=4,
            dim_output=1,
            dropout=0.3,
            type="sigmoid"
        )
        decoder3 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=256,
            num_layers=4,
            dim_output=33,
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

def build_vocabulary_dataset(data):
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
    return ds

def build_sentence_sanity_dataset(data):
    if os.path.exists("dataset/sentence_sanity.ds"):
        print("Loading sentence sanity dataset from cache.")
        return torch.load("dataset/sentence_sanity.ds")
    ds = []
    for item in tqdm(data, desc="Building sentence sanity dataset"):
        text : str = item["text"]
        remove_period = text.rstrip('.')
        if len(remove_period) == 0:
            continue
        y = torch.ones(len(remove_period), dtype=torch.float32)
        y[-1] = 0  # End token
        ds.append((1, as_utf32_tensor(remove_period), y))
    return ds

def build_cloze_dataset(data):
    if os.path.exists("dataset/cloze.ds"):
        print("Loading cloze dataset from cache.")
        return torch.load("dataset/cloze.ds")
    def randomly_remove_word(s: str):
        words = re.split(f"[{re.escape(string.punctuation)} \n\t]", s)
        if len(words) <= 1:
            return s  # Can't remove any word
        idx = random.randint(0, len(words) - 1)
        answer = words[idx]
        words[idx] = "<BLANK>"
        return " ".join(words), answer
    
    ds = []
    for item in tqdm(data, desc="Building cloze dataset"):
        sentence, answer = randomly_remove_word(item["text"])
        sentence = "<CLOZE TASK>: " + sentence
        # Add the 33rd dimension for end token
        answer_tensor = as_utf32_tensor(answer)
        answer_tensor = torch.cat([answer_tensor, torch.ones((1, 32), dtype=torch.float32)], dim=0)
        answer_tensor[-1] = torch.zeros(33, dtype=torch.float32)  # End token
        ds.append((2, as_utf32_tensor(sentence), ))
    return ds

def train(model: HRNN, decoder1: FFNNDecoder, decoder2: FFNNDecoder, decoder3: FFNNDecoder, data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(decoder1.parameters()) + list(decoder2.parameters()) + list(decoder3.parameters()), lr=0.001)

    model.train()
    decoder1.train()
    decoder2.train()
    decoder3.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.zero_grad()
    decoder1.zero_grad()
    decoder2.zero_grad()
    decoder3.zero_grad()
    
    print("Building datasets...")

    ds_word = build_vocabulary_dataset(data) * 10
    print("Building datasets 1/3")
    ds_sentence = build_sentence_sanity_dataset(data[:len(data) - 10000]) * 5
    print("Building datasets 2/3")
    ds_cloze = build_cloze_dataset(data[len(data) - 10000:]) * 2
    print("Building datasets 3/3, shuffling...")

    train = ds_word + ds_sentence + ds_cloze
    random.shuffle(train)
    losses = [-1, -1, -1]
    i = 0

    print("Start training process...")

    for task, x, y in tqdm(train):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        match task:
            case 0:
                y_hat = model(x, post_processing=decoder1)
            case 1:
                y_hat = model(x, post_processing=decoder2)
            case 2:
                y_hat = model.self_regression(x, max_output=64, halt_if=lambda t: (t.argmax(dim=-1) == 0).all(), post_processing=decoder3)
            case _:
                raise ValueError("Unknown task")

        # Compute loss
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
            })
        i += 1
        
        
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
if __name__ == "__main__":
    set_seed(114514)
    packed_models = load_checkpoint("checkpoints/word_identification.pth")
    train(*packed_models, ds["train"])