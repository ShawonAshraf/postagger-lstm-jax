# postagger-lstm-jax

A single layer LSTM part-of-speech tagger implemented in JAX (+Flax) on the `batterydata/pos_tagging` dataset
from Huggingface Datasets.

## Usage
Make sure that you have a wandb account and have logged in using your API key.
```bash
wandb login
```

Then run `main.py` with the following arguments:
```bash
python main.py --lr 0.01 --epochs 5 --batch-size 128 --seed 2023 --dropout 0.2 \
    --embedding-dim 300 --hidden-dim 300 --max_seq_len 300  \
    --pad_token_idx 1 --log_every_n_step 100
```

_The Trainer module is defined to train, evaluate and log to wandb simultaneously._

## Results
Check the wandb metrics [here](https://wandb.ai/shawonashraf/postagger-lstm-jax/runs/bs5n1ukb?workspace=user-shawonashraf).


## Environment Setup

__Version Requirements:__
- Python 3.11


```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --extra-index-url https://download.pytorch.org/whl/cpu
```