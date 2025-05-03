import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from cadre_algorithms.dataset import load_gene_expr_and_y, GDSCDataset
from cadre_algorithms.transformer import CADRETransformer
from cadre_algorithms.cadre import CADRE
from sklearn.metrics import roc_auc_score, f1_score
from datetime import datetime
import random
import os

def train(model, train_loader, test_loader, device, epochs=10, max_lr=1e-2, optimizer_type="adamw"):
    model.to(device)

    if optimizer_type.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=max_lr, momentum=0.9)
    else:
        optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.BCELoss()

    # Create a unique run folder under saved_models
    run_id = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("saved_models", run_id)
    os.makedirs(run_dir, exist_ok=True)

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
                if len(set(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_pred)
                    f1 = f1_score(y_true, (torch.tensor(y_pred) > 0.5).int())
                    print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
                else:
                    print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}, Skipped AUC (1 class)")

        print(f"[Epoch {epoch+1} Summary] Avg Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, test_loader, device)

        # Save checkpoint for this epoch
        ckpt_path = os.path.join(run_dir, f"epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")


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

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, (torch.tensor(y_pred) > 0.5).int())
        print(f"Test AUC: {auc:.4f}, F1: {f1:.4f}")
    else:
        print("Test: Skipped AUC — only one class in y_true.")


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
    train_genes = torch.stack([gene_expr_tensor[i] for i in train_idx])
    test_genes = torch.stack([gene_expr_tensor[i] for i in test_idx])

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
    gene_expr_path="original_code/data/input/exp_gdsc.csv",
    y_path="original_code/data/input/gdsc.csv",
    emb_dim=200,
    num_heads=1,
    num_layers=1,
    dropout=0.2,
    batch_size=16,
    epochs=10,
    max_lr=1e-2,
    pooling='mean',
    optimizer_type="adamw",
    model_type="transformer"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use:", device)

    gene_expr_tensor, y_tensor = load_gene_expr_and_y(gene_expr_path, y_path)
    gene_expr_tensor, y_tensor = load_gene_expr_and_y(gene_expr_path, y_path)

    # Check label validity
    bad_indices = torch.nonzero((y_tensor < 0) | (y_tensor > 1), as_tuple=False)
    print(f"Found {bad_indices.size(0)} invalid labels.")
    for idx in bad_indices[:10]:  # Show only first 10
        print(f"Label at {tuple(idx.tolist())} = {y_tensor[tuple(idx.tolist())].item()}")

    train_gene_expr, train_y_tensor, test_gene_expr, test_y_tensor = train_test_split_fixed(y_tensor, gene_expr_tensor)
    train_loader = DataLoader(GDSCDataset(train_gene_expr, train_y_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(GDSCDataset(test_gene_expr, test_y_tensor), batch_size=batch_size)

    # Dummy embedding for both models (replace if you load real embeddings)
    gene_emb_matrix = torch.nn.Embedding(2000, emb_dim).weight.detach()

    if model_type.lower() == "cadre":
        model = CADRE(gene_emb_tensor=gene_emb_matrix, num_pathways=train_y_tensor.shape[1], emb_dim=emb_dim, dropout=dropout)
    elif model_type.lower() == "transformer":
        model = CADRETransformer(
            gene_emb_matrix=gene_emb_matrix,
            num_drugs=train_y_tensor.shape[1],
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.to(device)
    print("Model on device:", next(model.parameters()).device)

    train(model, train_loader, test_loader, device, epochs=epochs, max_lr=max_lr, optimizer_type=optimizer_type)
