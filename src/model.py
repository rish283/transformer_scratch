import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # (batch size, sequence length, embedding dimension)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # size: (batch size, sequence length, sequence length)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1) 
        attn = self.dropout(attn)

        out = attn @ v # (batch size, sequence length, head dimension)
        return out 
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, block_size):
        super().__init__()
        head_dim = embed_dim //n_heads
        self.heads = nn.ModuleList([SelfAttention(embed_dim, head_dim, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        attn_out = torch.cat([head(self.ln1(x)) for head in self.heads], dim=-1)
        x = x + self.dropout(self.proj(attn_out))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, n_heads, n_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, block_size) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape # (batch size, sequence length)
        assert T <= self.block_size, "Input sequence length exceeds block size"
        token_embeddings = self.token_embed(idx)
        position_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1) # (batch size, sequence length)
        position_embeddings = self.pos_embed(position_ids) # (batch size, sequence length, embedding dimension)
        x = token_embeddings + position_embeddings
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) # (batch size, sequence length, vocab size)
        return logits
