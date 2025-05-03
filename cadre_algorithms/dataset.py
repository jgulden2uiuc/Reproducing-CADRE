import torch
from torch.utils.data import Dataset

class GDSCDataset(Dataset):
    def __init__(self, gene_expr_tensor, y_tensor):
        self.gene_expr_tensor = gene_expr_tensor
        self.y_tensor = y_tensor

        num_cells, num_drugs = y_tensor.shape
        self.pairs = [
            (i, j) for i in range(num_cells) for j in range(num_drugs)
            if y_tensor[i, j] != -1
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        gene_indices = self.gene_expr_tensor[i]  # Tensor of shape [1500]
        label = self.y_tensor[i, j]
        return gene_indices, j, label  # j is the drug ID
