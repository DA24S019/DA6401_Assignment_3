import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)

            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            targets_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, targets_flat)
            val_loss += loss.item()

            preds = output_flat.argmax(-1)
            mask = targets_flat != 0
            correct += (preds[mask] == targets_flat[mask]).sum().item()
            total += mask.sum().item()

    return val_loss / len(dataloader), correct / total
