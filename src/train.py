import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from src.data import load_data, get_dataloader
from src.models import DemodulatorMLP

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model(model_type="mlp", seed=42, epochs=20, lr=0.003, batch_size=128):
    set_seed(seed)
    use_polar = (model_type == "polar_mlp")
    
    X_train, y_train = load_data("train", use_polar=use_polar)
    X_val, y_val = load_data("val", use_polar=use_polar)
    
    input_dim = 4 if use_polar else 2
    model = DemodulatorMLP(input_dim=input_dim, hidden_dim=64, num_classes=16)
    
    loader = get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            
        train_loss = total_loss / len(X_train)
        
        # Validation SER
        val_preds = model.predict(X_val)
        val_ser = (val_preds != y_val).mean()
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"[{model_type} seed {seed}] Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - Val SER: {val_ser:.4f}")
            
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "polar_mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    train_model(args.model, seed=args.seed, epochs=args.epochs)
