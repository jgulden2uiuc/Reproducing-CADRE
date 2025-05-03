import torch
import torch.nn as nn
import torch.nn.functional as F

class CADRETransformer(nn.Module):
    def __init__(self, gene_emb_matrix, num_drugs,
                 emb_dim=200, num_heads=8, num_layers=4,
                 dropout=0.2, pooling='mean'):
        super().__init__()

        assert pooling in ['mean', 'max'], "pooling must be 'mean' or 'max'"
        self.pooling = pooling

        # Frozen pretrained gene embeddings
        self.gene_emb = nn.Embedding.from_pretrained(gene_emb_matrix, freeze=True)

        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True  # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable drug embeddings
        self.drug_emb = nn.Embedding(num_drugs, emb_dim)

        self.final_dropout = nn.Dropout(dropout)

    def forward(self, gene_indices, drug_ids):
        gene_embed = self.gene_emb(gene_indices)   # [B, 1500, D]
        encoded = self.encoder(gene_embed)         # [B, 1500, D]

        # Pooling to fixed-size cell line embedding
        if self.pooling == 'mean':
            cell_embed = encoded.mean(dim=1)       # [B, D]
        elif self.pooling == 'max':
            cell_embed = encoded.max(dim=1).values # [B, D]

        cell_embed = self.final_dropout(cell_embed)

        drug_embed = self.drug_emb(drug_ids)       # [B, D]

        # Dot product prediction
        dot = torch.sum(cell_embed * drug_embed, dim=1)  # [B]
        return torch.sigmoid(dot)
