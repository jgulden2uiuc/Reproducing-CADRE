import torch
import torch.nn as nn
import torch.nn.functional as F

class CADRE(nn.Module):
    def __init__(self, gene_emb_tensor, num_pathways, emb_dim=200, dropout=0.2):
        super().__init__()
        self.gene_emb = nn.Embedding.from_pretrained(gene_emb_tensor, freeze=True)
        self.pathway_emb = nn.Embedding(num_pathways, emb_dim)
        self.drug_emb = nn.Parameter(torch.randn(num_pathways, emb_dim))

        self.attn_tanh = nn.Linear(emb_dim, emb_dim)
        self.attn_query = nn.Linear(emb_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_indices_batch, pathway_indices):
        B, K = gene_indices_batch.shape  # B=batch size, K=top-k genes

        gene_emb = self.gene_emb(gene_indices_batch)              # (B, K, D)
        pathway_emb = self.pathway_emb(pathway_indices).unsqueeze(1)  # (B, 1, D)

        attn_input = torch.tanh(self.attn_tanh(gene_emb + pathway_emb))  # (B, K, D)
        attn_scores = self.attn_query(attn_input).squeeze(-1)            # (B, K)
        attn_weights = F.softmax(attn_scores, dim=-1)                    # (B, K)

        context = torch.sum(attn_weights.unsqueeze(-1) * gene_emb, dim=1)  # (B, D)
        context = self.dropout(context)

        drug_emb = self.drug_emb[pathway_indices]  # (B, D)
        dot = torch.sum(context * drug_emb, dim=1)  # (B,)
        pred = torch.sigmoid(dot)
        return pred
