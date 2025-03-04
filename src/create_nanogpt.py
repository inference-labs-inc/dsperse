# create_nanogpt.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class NanoGPT(nn.Module):
    """
    A simplified version of a nanoGPT model.
    """

    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super(NanoGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 100, n_embd)
        )  # Assume max 100 positions
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        # x: (batch, seq_length)
        token_emb = self.token_embedding(x)
        seq_length = x.size(1)
        pos_emb = self.position_embedding[:, :seq_length, :]
        x = token_emb + pos_emb
        x = x.transpose(0, 1)  # Transformer expects (seq_length, batch, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def create_nanogpt(model_params: dict) -> nn.Module:
    """
    Create a nanoGPT model based on provided parameters.

    Args:
        model_params (dict): Dictionary containing model configuration options.

    Returns:
        nn.Module: A nanoGPT model instance.
    """
    model = NanoGPT(
        vocab_size=model_params.get("vocab_size", 50257),
        n_embd=model_params.get("n_embd", 64),
        n_layer=model_params.get("n_layer", 2),
        n_head=model_params.get("n_head", 4),
    )
    return model


def train_nanogpt(model: nn.Module, epochs: int = 1, batch_size: int = 2):
    """
    Train the nanoGPT model on dummy data.

    Args:
        model (nn.Module): The nanoGPT model to be trained.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.

    Returns:
        None
    """
    # Generate dummy data
    input_data = torch.randint(0, 50257, (100, 10))  # 100 samples, sequence length 10
    target_data = torch.randint(0, 50257, (100, 10))

    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # Flatten the outputs and targets for the loss function
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


