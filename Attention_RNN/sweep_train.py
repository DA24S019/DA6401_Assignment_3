import torch
import torch.nn as nn
import wandb
from model import Encoder, Decoder, Seq2Seq, ModelConfig
from data import load_data  # assumes load_data returns tokenizers and loaders

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            self.counter = 0
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: EarlyStopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping triggered')
                self.early_stop = True


def train():
    wandb.init()
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, input_tokenizer, output_tokenizer = load_data(config.batch_size)
    
    model_config = ModelConfig(
        input_vocab_size=input_tokenizer.vocab_size,
        output_vocab_size=output_tokenizer.vocab_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        cell_type=config.cell_type,
        dropout=config.dropout
    )
    
    encoder = Encoder(model_config)
    decoder = Decoder(model_config)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_acc = 0.0
    early_stopper = EarlyStopping(patience=5, min_delta=1e-4)
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, config.teacher_forcing)

            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            targets_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, targets_flat)

            predictions = output_flat.argmax(-1)
            mask = targets_flat != 0
            train_correct += (predictions[mask] == targets_flat[mask]).sum().item()
            train_total += mask.sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'input_tokenizer': input_tokenizer.char2idx,
                'output_tokenizer': output_tokenizer.char2idx,
                'config': model_config.__dict__
            }, 'best_model_attention.pth')
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc
        })

        print(f"Epoch {epoch+1:03} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        early_stopper(val_acc)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1:03}")
            break


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)

            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            targets_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()

            predictions = output.argmax(-1)
            mask = trg != 0
            correct += (predictions[mask] == trg[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy
