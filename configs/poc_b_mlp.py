from nanolgn.config import TransformerCfg

cfg = TransformerCfg(
    d_model=256, n_layer=6, n_head=4, ctx_len=512,
    vocab_size=50257, ffn="mlp", seed=0,
)
lgn = None

batch_size = 16
max_steps = 20000
warmup_steps = 200
peak_lr = 1e-4
min_lr = 1e-5
weight_decay = 0.1
grad_clip = 1.0
eval_every = 250
eval_tokens = 1_000_000
log_every = 50
log_block_stats_until = 200
data_train = "data/fineweb_train.bin"
data_val   = "data/fineweb_val.bin"
