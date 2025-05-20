import torch
import torch.nn as nn
import wandb

from model import Encoder, Decoder, Seq2Seq
from config import ModelConfig
from loader import load_data
from evaluate import evaluate

def train():
    wandb.init()
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     # Load data
    train_loader, val_loader, input_tokenizer, output_tokenizer = load_data(config.batch_size)

    # Model configuration
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

    # Initialize model
    # Build encoder & decoder from your config
    encoder = Encoder(model_config)
    decoder = Decoder(model_config)

# Now correctly instantiate your seq2seq model
    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 5  # Early stopping

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
            epochs_no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'input_tokenizer': input_tokenizer.char2idx,
                'output_tokenizer': output_tokenizer.char2idx,
                'config': model_config.__dict__
            }, 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

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
