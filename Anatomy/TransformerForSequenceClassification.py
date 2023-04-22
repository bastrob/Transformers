from torch import nn

from TransformerEncoder import TransformerEncoder


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]  # select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x
