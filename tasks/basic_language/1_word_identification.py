import torch
from torch import nn, optim
import os
import string

from model.model import HRNN
from model.decoder import FFNNDecoder

from datasets import load_dataset

ds = load_dataset("agentlans/high-quality-english-sentences")

def load_checkpoint(checkpoint_path: str):
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
            dim_output=32,
            dropout=0.3,
            type="softmax"
        )
    else:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        decoder1 = checkpoint["decoder1"]
        decoder2 = checkpoint["decoder2"]
    return model, decoder1, decoder2

def as_utf32_tensor(s: str):
    # To 32-bit 1-hot tensor
    tensor = torch.zeros(len(s), 32, dtype=torch.int32)
    for i, c in enumerate(s):
        tensor[i, ord(c)] = 1
    return tensor

def build_vocabulary_dataset(data):
    vocab = set()
    ds = []
    for item in data:
        text : str = item['text']
        words = text.split(string.punctuation + " \n\t")
        vocab.update(words)
    for v in vocab:
        y = torch.zeros(len(v), dtype=torch.int32)
        y[-1] = 1  # End token
        ds.append((as_utf32_tensor(v), 1))
    return ds

def word_sanity(model: HRNN, decoder1: FFNNDecoder, data, device):
    for x, y in data:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        sanity = decoder1(output)
        yield sanity, y
        
def cloze(model: HRNN, decoder2: FFNNDecoder, data, device):
    for x, y in data:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        cloze = decoder2(output)
        yield cloze, y

def train(model: HRNN, decoder1: FFNNDecoder, decoder2: FFNNDecoder, data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(decoder1.parameters()) + list(decoder2.parameters()), lr=0.001)
    
    model.train()
    decoder1.train()
    decoder2.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.zero_grad()
    decoder1.zero_grad()
    decoder2.zero_grad()

    for batch in data:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_hat = model(x)
        y_hat = decoder1(y_hat)

        # Compute loss
        loss = loss_fn(y_hat, y)

        # Backward pass
        loss.backward()

        optimizer.step()
        
    