from cadre_algorithms.train_utils import run_transformer_pipeline

run_transformer_pipeline(
    batch_size=4,
    epochs=1,
    max_lr=1e-2,
    dropout=0.2,
    optimizer_type='adamw'
)
