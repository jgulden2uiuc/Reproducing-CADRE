# CADRE: Contextual Attention-Based Drug Response Model

This repository contains a PyTorch reimplementation and extension of the CADRE model from Tao et al. (2020), which predicts binary drug sensitivity in cancer cell lines using contextual attention over gene embeddings. We also provide a transformer-based variant for comparison.

## Contents

- `cadre.py` – Original CADRE model with contextual attention over gene embeddings.
- `transformer.py` – Transformer-based extension using self-attention.
- `dataset.py` – Dataset loading and preprocessing utilities.
- `train_utils.py` – Training and evaluation pipeline.
- `__init__.py` – Exports module components.

## How to Use

### 1. Colab Notebook

To use this code on **Google Colab**:

1. Upload the entire repository to your Google Drive.
2. Open a new Colab notebook.
3. Mount Google Drive using:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Navigate to the directory and run:
   ```python
   %cd /content/drive/MyDrive/your_repo_folder
   !pip install -r requirements.txt  # Probably not necessary; simply pip install requirements as needed
   ```
5. Use the training pipeline:
   ```python
   from cadre_algorithms.train_utils import run_transformer_pipeline
   run_transformer_pipeline()
   ```

This will train a model on the GDSC dataset and log performance per epoch.

### 2. Local GPU Setup (Recommended)

Ensure you have:
- Python ≥ 3.7
- PyTorch with CUDA support (for GPU acceleration)

Install dependencies (if not using a virtual environment, prefix with `pip`):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas scikit-learn matplotlib umap-learn
```

Then run the pipeline from `train_utils.py`:

```python
from cadre_algorithms.train_utils import run_transformer_pipeline
run_transformer_pipeline(model_type="cadre")  # or "transformer"
```

Alternatively run "python main.py", with `main.py` edited with parameters of your choosing.

## What the Code Does

This repo reproduces the original CADRE model and evaluates an extension using transformer encoders:

- The **CADRE model** (see `cadre.py`) uses a contextual attention mechanism over top-k gene embeddings, influenced by the target drug’s pathway, to compute a dot-product with a learnable drug embedding.
- The **Transformer Encoder model** (see `transformer.py`) replaces this with a self-attention encoder (no explicit pathway embedding), inspired by NLP transformers.
- The training loop (see `train_utils.py`) loads binary drug response labels and gene expression profiles from CSVs, splits them into training and test sets, and trains the model using binary cross-entropy on observed pairs only.
- Metrics include AUROC and F1, logged per epoch.

The full training process:
1. Loads `exp_gdsc.csv` (gene expressions) and `gdsc.csv` (drug response labels).
2. Builds train/test loaders using only valid (non-missing) cell-drug pairs.
3. Initializes either CADRE or transformer model.
4. Trains using `OneCycleLR` scheduler and `AdamW` or `SGD` optimizer.
5. Logs loss, AUC, and F1 every 100 steps.
6. Saves epoch-wise checkpoints to `saved_models/{model_name}_{timestamp}/`.

## Outputs You Will See

While running, the following outputs will appear:

- Device in use (e.g. `cuda` if GPU is enabled)
- Step-by-step training loss, AUC, and F1 scores
- Evaluation metrics at the end of each epoch
- Checkpoint paths for each saved model

Example:
```
[Epoch 1 | Step 0] Loss: 0.6931, AUC: 0.524, F1: 0.487
[Epoch 1 Summary] Avg Loss: 0.6821
Test AUC: 0.7924, F1: 0.6153
Checkpoint saved: saved_models/CADRE_20250503_183210/epoch1.pth
```
Note that the numbers after CADRE (or CADRETransformer) are the formatted datetime

## Data Used

- `gdsc.csv` – binary drug sensitivity labels (0 = resistant, 1 = sensitive)
- `exp_gdsc.csv` – binary gene expression indicators (for top-1500 genes)
- `exp_emd_gdsc.csv` – 200D pretrained gene embeddings (not currently loaded)
- `drug_info_gdsc.csv` – drug metadata (optional for analysis)

Ensure these files are in `original_code/data/input/` or change the paths in `run_transformer_pipeline()`.

## Notes

- GPU strongly recommended. CPU training can take 6+ hours.
- For rapid testing, reduce `epochs`, `batch_size`, and increase `max_lr`.
- Model outputs are saved to `saved_models/` after every epoch.
- The default model uses a dummy embedding matrix. Replace with real embeddings for best performance.

## Report Summary

This codebase was created for a reproducibility study of the CADRE model. As described in the final report:

- The CADRE model is robust and achieved AUC ≈ 0.83 and F1 ≈ 0.64, matching the original paper.
- Transformer-based models (our extension) achieved slightly lower performance, but validated the feasibility of alternative attention architectures.
- Evaluation is performed using binary classification metrics with masking for missing drug responses.

See the final report for detailed benchmarks and experiment logs.

---

Maintainers: John Gulden, Varun Lagadapati (University of Illinois at Urbana-Champaign)
