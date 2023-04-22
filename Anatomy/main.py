from transformers import AutoTokenizer
from torch import nn
from transformers import AutoConfig

from AttentionHead import AttentionHead
from Embeddings import Embeddings
from MultiHeadAttention import MultiHeadAttention
from TransformerEncoder import TransformerEncoder
from TransformerEncoderLayer import TransformerEncoderLayer
from TransformerForSequenceClassification import TransformerForSequenceClassification

text = "time flies like an arrow"
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

heads = MultiHeadAttention(config)
print(heads(inputs_embeds).size())

encoder_layer = TransformerEncoderLayer(config)
print(encoder_layer(inputs_embeds).size())

emb = Embeddings(config)
print(emb(inputs.input_ids).size())


encoder = TransformerEncoder(config)
print(encoder(inputs.input_ids).size())

config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
print(encoder_classifier(inputs.input_ids).size())