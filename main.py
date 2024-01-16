import wandb
from model.tagger import LSTMTagger
from jax import random
import numpy as np

from data.utils import preprocess
from data.dataset import TagDataset
from datasets import load_dataset
import jax_dataloader as jdl

from training.trainer import Trainer

import argparse

parser = argparse.ArgumentParser(description="Train a LSTM tagger on the batterydata/pos_tagging dataset")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimiser")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
parser.add_argument("--seed", type=int, default=2023, help="seed for reproducibility")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate for the LSTM")
parser.add_argument("--embedding_dim", type=int, default=300, help="embedding dimension for the LSTM")
parser.add_argument("--hidden_dim", type=int, default=300, help="hidden dimension for the LSTM")
parser.add_argument("--max_seq_len", type=int, default=300, help="maximum sequence length to pad all sequences to")
parser.add_argument(("--pad_token_idx"), type=int, default=1, help="padding token index in the vocabulary")
parser.add_argument("--log_every_n_step", type=int, default=100, help="log every n steps")

args = parser.parse_args()

if __name__ == "__main__":
    # dataset
    main_dataset = load_dataset("batterydata/pos_tagging")
    word_to_idx, label_to_idx, train_split, validation_split, test_split = preprocess(main_dataset,
                                                                                      pad_token_idx=args.pad_token_idx,
                                                                                      validation_size=0.2, seed=2023)

    dataset_config = {
        "pad_token_idx": args.pad_token_idx,
        "max_seq_len": args.max_seq_len,
        "word_to_idx": word_to_idx,
        "label_to_idx": label_to_idx
    }

    train_set = TagDataset(train_split, **dataset_config)
    val_set = TagDataset(validation_split, **dataset_config)
    test_set = TagDataset(test_split, **dataset_config)

    # dataloader
    train_loader = jdl.DataLoader(train_set, "pytorch", batch_size=args.batch_size, shuffle=True)
    val_loader = jdl.DataLoader(val_set, "pytorch", batch_size=args.batch_size, shuffle=False)
    test_loader = jdl.DataLoader(test_set, "pytorch", batch_size=args.batch_size, shuffle=False)

    # model
    # model prng
    master_key = random.key(seed=args.seed)
    master_key, model_init_key = random.split(master_key)
    master_key, dropout_key = random.split(master_key)
    model_rngs = {"params": model_init_key, "dropout": dropout_key}

    model_config = {
        "vocab_size": len(word_to_idx),
        "embedding_dimensions": args.embedding_dim,
        "lstm_hidden_dims": args.hidden_dim,
        "n_labels": args.max_seq_len,
        "lstm_seed": args.seed + 1,
        "dropout_rate": args.dropout,
    }

    # init params
    model = LSTMTagger(**model_config)
    init_params = model.init(model_rngs, np.arange(args.max_seq_len))

    # logger
    logger_config = {
        "project": "postagger-lstm-jax",
        "config": {
            "learning_rate": args.lr,
            "optimiser": "adam",
            "architecture": "LSTM",
            "dataset": "batterydata/pos_tagging",
            "pad_token_idx": args.pad_token_idx,
            "max_seq_len": args.max_seq_len,
            "vocab_size": len(word_to_idx),
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "dropout": args.dropout,
            "epochs": args.epochs,
        }
    }
    wandb.init(**logger_config)

    # trainer
    trainer_config = {
        "model": model,
        "init_params": init_params,
        "lr": args.lr,
        "max_epochs": args.epochs,
        "log_every_n_step": args.log_every_n_step,
        "model_rngs": model_rngs,
        "logger": wandb
    }

    trainer = Trainer(**trainer_config)

    state = trainer.fit_and_eval(train_loader, val_loader, test_loader)

    # finish logging
    wandb.finish()
