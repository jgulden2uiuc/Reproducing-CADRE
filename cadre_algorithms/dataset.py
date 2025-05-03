import torch
import pandas as pd
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
        gene_indices = self.gene_expr_tensor[i]
        label = self.y_tensor[i, j].item()  # ensure it's a Python float

        if not (0.0 <= label <= 1.0):
            print(f"Label out of bounds at ({i}, {j}):", label)
            label = 0.0

        return gene_indices, j, label


def load_gene_expr_and_y(expr_path, label_path):
    gene_expr_df = pd.read_csv(expr_path, index_col=0)
    label_df = pd.read_csv(label_path, index_col=0)

    label_df = label_df.fillna(0)

    gene_expr_tensor = torch.tensor(gene_expr_df.values, dtype=torch.long)
    y_tensor = torch.tensor(label_df.values, dtype=torch.float)
    return gene_expr_tensor, y_tensor
