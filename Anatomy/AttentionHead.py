import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) // sqrt(dim_k)
    if mask is not None:
        scores = torch.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )
        return attn_outputs
