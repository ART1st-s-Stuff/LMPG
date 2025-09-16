import torch
from torch import nn, optim
import os

from model.model import HRNN
from model.decoder import FFNNDecoder

def data_parser()

def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        model = HRNN.default()
        decoder = FFNNDecoder(
            dim_input=model.dim_output,
            dim_hidden=256,
            num_layers=4,
            dim_output=1,
            dropout=0.3,
            type="sigmoid"
        )
    else:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        decoder = checkpoint["decoder"]
    return model, decoder

def train(model: HRNN, decoder: FFNNDecoder, data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=0.001)
    
    model.train()
    decoder.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.zero_grad()
    decoder.zero_grad()

    for batch in data:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_hat = model(x)
        y_hat = decoder(y_hat)

        # Compute loss
        loss = loss_fn(y_hat, y)

        # Backward pass
        loss.backward()

        optimizer.step()
        
    