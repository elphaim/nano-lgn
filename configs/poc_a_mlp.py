"""POC-A baseline: standard ReLU² MLP, d_model=128, 4 layers."""
from nanolgn.config import TransformerCfg

cfg = TransformerCfg(
    d_model=128, n_layer=4, n_head=4, ctx_len=256,
    vocab_size=50257, ffn="mlp", seed=0,
)
lgn = None

# Training schedule.
batch_size = 32
max_steps = 5000
warmup_steps = 200
peak_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
grad_clip = 1.0
eval_every = 250
eval_tokens = 1_000_000
log_every = 50
log_block_stats_until = 200
data_train = "data/tinystories_train.bin"
data_val   = "data/tinystories_val.bin"
