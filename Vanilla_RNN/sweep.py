# sweep_runner.py
import wandb
from train_sweep import train

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "embedding_size": {"values": [16, 32, 64, 128, 256]},
        "hidden_size": {"values": [16, 32, 64, 128, 256]},
        "encoder_layers": {"values": [1, 2, 3]},
        "decoder_layers": {"values": [1, 2, 3]},
        "cell_type": {"values": ["rnn", "gru", "lstm"]},
        "dropout": {"values": [0.2, 0.3]},
        "batch_size": {"values": [64, 128]},
        "learning_rate": {"values": [0.001, 0.0005]},
        "teacher_forcing": {"values": [0.5, 0.7]},
        "epochs": {"values": [15, 20]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_3")
    wandb.agent(sweep_id, function=train, count=50)
