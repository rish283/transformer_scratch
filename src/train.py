import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, EPOCHS, train_loader, val_loader, loss_fn, optimizer, device):
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(EPOCHS):
        # Training
        total_loss = 0.0
        model.train()
        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{EPOCHS}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                y = y.view(B * T)
                val_loss = loss_fn(logits, y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss:.4f}")
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
    # Save the best model to a file
    torch.save(best_model, 'best_model.pth')
    print("Training complete. Best model saved.")
