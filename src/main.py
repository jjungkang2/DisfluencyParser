import argparse
import itertools
import os.path
import time
import sys
import torch
import torch.optim.lr_scheduler

import numpy as np
import random

import vocabulary
import nkutil
import parse_nk

def torch_load(load_path):
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def make_hparams():
    return nkutil.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        sentence_max_len=300,

        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True,
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3, # establishes a termination criterion

        partitioned=True,
        num_layers_position_only=0,

        num_layers=8,
        d_model=1024,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_tag_hidden=250,
        tag_loss_scale=5.0,

        attention_dropout=0.2,
        embedding_dropout=0.0,
        relu_dropout=0.1,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_bert=False,
        predict_tags=False,

        d_char_emb=32,

        tag_emb_dropout=0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,

        bert_model="bert-base-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        )

#%%

def run_parse(args):
    
    print("Loading model from {}...".format(args.model_path_base))

    info = torch_load(args.model_path_base)
    parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

    print("Parsing sentences...")
    with open(args.input_path) as input_file:
        sentences = input_file.readlines()
    sentences = [sentence.split() for sentence in sentences]

    # Tags are not available when parsing from raw text, so use a dummy tag
    if 'UNK' in parser.tag_vocab.indices:
        dummy_tag = 'UNK'
    else:
        dummy_tag = parser.tag_vocab.value(0)

    start_time = time.time()

    all_predicted = []
    for start_index in range(0, len(sentences), args.eval_batch_size):
        subbatch_sentences = sentences[start_index:start_index+args.eval_batch_size]

        subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
        predicted, _ = parser.parse_batch(subbatch_sentences)
        del _
        if args.output_path == '-':
            for p in predicted:
                print(p.convert().linearize())
        else:
            all_predicted.extend([p.convert() for p in predicted])

    if args.output_path != '-':
        with open(args.output_path, 'w') as output_file:
            for tree in all_predicted:
                output_file.write("{}\n".format(tree.linearize()))
        print("Output written to:", args.output_path)
#%%

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("parse")
    subparser.set_defaults(callback=run_parse)
    subparser.add_argument("--model-path-base", default="best_models/swbd_fisher_bert_Edev.0.9078.pt")
    subparser.add_argument("--input-path", default="best_models/raw_sentences.txt")
    subparser.add_argument("--output-path", default="best_models/parsed_sentences.txt")
    subparser.add_argument("--eval-batch-size", type=int, default=100)

    args = parser.parse_args()
    args.callback(args)

# %%
if __name__ == "__main__":
    main()
