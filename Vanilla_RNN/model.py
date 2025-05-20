import torch
import torch.nn as nn
from config import ModelConfig
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.input_vocab_size, config.embedding_size)
        rnn_map = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.rnn = rnn_map[config.cell_type](
            config.embedding_size, config.hidden_size,
            num_layers=config.encoder_layers,
            dropout=config.dropout if config.encoder_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.output_vocab_size, config.embedding_size)
        rnn_map = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.rnn = rnn_map[config.cell_type](
            config.embedding_size, config.hidden_size,
            num_layers=config.decoder_layers,
            dropout=config.dropout if config.decoder_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, config.output_vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        return self.fc(output), hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        outputs = torch.zeros(batch_size, trg_len, self.decoder.fc.out_features).to(self.device)
        
        # Get encoder's hidden state
        encoder_hidden = self.encoder(src)
        
        # Adjust hidden state dimensions for decoder
        if isinstance(encoder_hidden, tuple):  # LSTM case
            h, c = encoder_hidden
            decoder_layers = self.decoder.rnn.num_layers
            
            # Adjust hidden states
            h = self._adjust_hidden(h, decoder_layers)
            c = self._adjust_hidden(c, decoder_layers)
            decoder_hidden = (h, c)
        else:  # GRU/RNN case
            decoder_hidden = self._adjust_hidden(encoder_hidden, self.decoder.rnn.num_layers)
        
        # Decoder initialization
        input = trg[:, 0].unsqueeze(1)
        
        # Decoder forward with teacher forcing
        for t in range(1, trg_len):
            output, decoder_hidden = self.decoder(input, decoder_hidden)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t].unsqueeze(1) if teacher_force else output.argmax(-1)
            
        return outputs

    def _adjust_hidden(self, hidden, target_layers):
        current_layers = hidden.size(0)
        if current_layers == target_layers:
            return hidden
        elif current_layers < target_layers:
            # Repeat last layer to match decoder layers
            diff = target_layers - current_layers
            last_layer = hidden[-1:, :, :]
            return torch.cat([hidden, last_layer.repeat(diff, 1, 1)], dim=0)
        else:
            # Trim extra layers
            return hidden[:target_layers]