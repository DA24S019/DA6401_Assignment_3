from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_vocab_size: int
    output_vocab_size: int
    embedding_size: int
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    cell_type: str
    dropout: float

