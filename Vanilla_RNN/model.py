import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden[-1]
        decoder_hidden = decoder_hidden.unsqueeze(1)
        scores = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden)))
        attn_weights = F.softmax(scores, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.input_vocab_size, config.embedding_size)
        rnn_map = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.rnn = rnn_map[config.cell_type](
            config.embedding_size,
            config.hidden_size,
            num_layers=config.encoder_layers,
            dropout=config.dropout if config.encoder_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.output_vocab_size, config.embedding_size)
        self.attention = BahdanauAttention(config.hidden_size)
        rnn_map = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.rnn = rnn_map[config.cell_type](
            config.embedding_size + config.hidden_size,
            config.hidden_size,
            num_layers=config.decoder_layers,
            dropout=config.dropout if config.decoder_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, config.output_vocab_size)

    def forward(self, x, hidden, encoder_outputs, return_attention=False):
        embedded = self.embedding(x)
        attn_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        context, attn_weights = self.attention(encoder_outputs, attn_hidden)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output)
        return (prediction, hidden, attn_weights) if return_attention else (prediction, hidden)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _adjust_hidden(self, hidden, target_layers):
        if hidden.size(0) == target_layers:
            return hidden
        elif hidden.size(0) < target_layers:
            diff = target_layers - hidden.size(0)
            return torch.cat([hidden, hidden[-1:].repeat(diff, 1, 1)], dim=0)
        else:
            return hidden[:target_layers]

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        encoder_outputs, encoder_hidden = self.encoder(src)

        if isinstance(encoder_hidden, tuple):
            h, c = encoder_hidden
            h = self._adjust_hidden(h, self.decoder.rnn.num_layers)
            c = self._adjust_hidden(c, self.decoder.rnn.num_layers)
            decoder_hidden = (h, c)
        else:
            decoder_hidden = self._adjust_hidden(encoder_hidden, self.decoder.rnn.num_layers)

        outputs = torch.zeros(batch_size, trg_len, self.decoder.fc.out_features).to(self.device)
        input = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, decoder_hidden = self.decoder(input, decoder_hidden, encoder_outputs)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t].unsqueeze(1) if teacher_force else output.argmax(-1)

        return outputs
