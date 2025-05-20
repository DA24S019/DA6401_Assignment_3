import wandb
from sweep_train import train

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [15, 20]},
        'batch_size': {'values': [32, 64]},
        'embedding_size': {'values': [64, 128]},
        'hidden_size': {'values': [128, 256]},
        'encoder_layers': {'values': [1, 2]},
        'decoder_layers': {'values': [1, 2]},
        'dropout': {'values': [0.2, 0.3]},
        'teacher_forcing': {'values': [0.5, 1.0]},
        'cell_type': {'values': ['gru', 'lstm']}
    }
}

def sweep_train():
    train()  # uses wandb.config internally

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='dakshina-attention-transliteration')
    wandb.agent(sweep_id, function=sweep_train)
