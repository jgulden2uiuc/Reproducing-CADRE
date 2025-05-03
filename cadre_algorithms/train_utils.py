import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from cadre_models.dataset import load_gene_expr_and_y, GDSCDataset
from cadre_models.train_utils import train_test_split_fixed, train
from cadre_models.transformer import CADRETransformer
from sklearn.metrics import roc_auc_score, f1_score
import random
import torch

def train(model, train_loader, test_loader, device, epochs=10, max_lr=1e-2, optimizer_type="adamw"):
    model.to(device)

    if optimizer_type.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=max_lr, momentum=0.9)
    else:
        optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        y_true, y_pred = [], []

        for step, batch in enumerate(train_loader):
            genes, conds, labels = batch
            genes = genes.to(device)
            conds = conds.to(device)
            labels = labels.float().to(device)

            preds = model(genes, conds)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

            if step % 100 == 0:
                auc = roc_auc_score(y_true, y_pred)
                f1 = f1_score(y_true, (torch.tensor(y_pred) > 0.5).int())
                print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

        print(f"[Epoch {epoch+1} Summary] Avg Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, test_loader, device)


def evaluate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in data_loader:
            genes, conds, labels = batch
            genes = genes.to(device)
            conds = conds.to(device)
            preds = model(genes, conds)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, (torch.tensor(y_pred) > 0.5).int())
    print(f"Test AUC: {auc:.4f}, F1: {f1:.4f}")


def train_test_split_fixed(y_tensor, gene_expr_tensor, extra_tensor=None, test_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    num_cells = y_tensor.shape[0]
    indices = list(range(num_cells))
    random.shuffle(indices)

    split = int(num_cells * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_y = y_tensor[train_idx]
    test_y = y_tensor[test_idx]
    train_genes = [gene_expr_tensor[i] for i in train_idx]
    test_genes = [gene_expr_tensor[i] for i in test_idx]

    if extra_tensor is not None:
        return train_genes, train_y, extra_tensor, test_genes, test_y, extra_tensor
    else:
        return train_genes, train_y, test_genes, test_y


def align_y_and_pathways(y_tensor, drug_pathway_tensor):
    if y_tensor.shape[1] > drug_pathway_tensor.shape[0]:
        print(f"Dropping {y_tensor.shape[1] - drug_pathway_tensor.shape[0]} unmatched drug(s) from y_tensor")
        y_tensor = y_tensor[:, :drug_pathway_tensor.shape[0]]
    elif drug_pathway_tensor.shape[0] > y_tensor.shape[1]:
        print(f"Dropping {drug_pathway_tensor.shape[0] - y_tensor.shape[1]} unmatched pathways")
        drug_pathway_tensor = drug_pathway_tensor[:y_tensor.shape[1]]

    assert y_tensor.shape[1] == drug_pathway_tensor.shape[0], "Still mismatched after fixing!"
    return y_tensor, drug_pathway_tensor


def run_transformer_pipeline(
    gene_expr_path="original_code/data/input/exp_idx_gdsc.csv",
    y_path="original_code/data/input/gdsc.csv",
    emb_dim=200,
    num_heads=1,
    num_layers=4,
    dropout=0.2,
    batch_size=64,
    epochs=10,
    max_lr=1e-2,
    pooling='mean',
    optimizer_type="adamw"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_expr_tensor, y_tensor = load_gene_expr_and_y(gene_expr_path, y_path)
    train_gene_expr, train_y_tensor, test_gene_expr, test_y_tensor = train_test_split_fixed(y_tensor, gene_expr_tensor)
    train_loader = DataLoader(GDSCDataset(train_gene_expr, train_y_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(GDSCDataset(test_gene_expr, test_y_tensor), batch_size=batch_size)

    gene_emb_matrix = torch.nn.Embedding(2000, emb_dim).weight.detach()  # dummy
    model = CADRETransformer(
        gene_emb_matrix=gene_emb_matrix,
        num_drugs=train_y_tensor.shape[1],
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling
    )

    train(model, train_loader, test_loader, device, epochs=epochs, max_lr=max_lr, optimizer_type=optimizer_type)
