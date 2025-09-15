import torch
from torch import nn, optim
import os
import string
import random
import tqdm

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
            dim_output=1,
            dropout=0.3,
            type="softmax"
        )
        decoder3 = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=256,
            num_layers=4,
            dim_output=33,
            dropout=0.3,
            type="softmax"
        )
    else:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        decoder1 = checkpoint["decoder1"]
        decoder2 = checkpoint["decoder2"]
        decoder3 = checkpoint["decoder3"]
    return model, decoder1, decoder2, decoder3

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
        y = torch.ones(len(v), dtype=torch.int32)
        y[-1] = 0  # End token
        ds.append((0, as_utf32_tensor(v), y))
    return ds

def build_sentence_sanity_dataset(data):
    ds = []
    for item in data:
        text : str = item['text']
        remove_period = text.rstrip('.')
        y = torch.ones(len(remove_period), dtype=torch.int32)
        y[-1] = 0  # End token
        ds.append((1, as_utf32_tensor(remove_period), y))
    return ds

def build_cloze_dataset(data):
    def randomly_remove_word(s: str):
        words = s.split(string.punctuation + " \n\t")
        if len(words) <= 1:
            return s  # Can't remove any word
        idx = random.randint(0, len(words) - 1)
        answer = words[idx]
        words[idx] = "<BLANK>"
        return " ".join(words), answer
    
    ds = []
    for item in data:
        sentence, answer = randomly_remove_word(item['text'])
        sentence = "<CLOZE TASK>: " + sentence
        ds.append((2, as_utf32_tensor(sentence), as_utf32_tensor(answer)))
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

    ds_word = build_vocabulary_dataset(data) * 20
    ds_sentence = build_sentence_sanity_dataset(data[:-10000]) * 10
    ds_cloze = build_cloze_dataset(data[-10000:]) * 5

    train = ds_word + ds_sentence + ds_cloze
    random.shuffle(train)
    losses = [-1, -1, -1]
    i = 0

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
    train(*packed_models, ds)