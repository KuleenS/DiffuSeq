"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb

from dotenv import load_dotenv
from TubeViT.tubevit.model import TubeViTLightningModule
import torch.nn as nn

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
# os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    load_dotenv()
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    args.batch_size = 1
    args.microbatch = 1

    args.learning_steps = 1000
    args.data_folder = "/scratch/kxu39/3videos"

    print(f"args: {args}")
    # dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        folder = args.data_folder,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight # use model's weights as init
    )
    
    next(data)

    # data_valid = load_data_text(
    #     batch_size=args.batch_size,
    #     seq_len=args.seq_len,
    #     data_args=args,
    #     folder = args.data_folder,
    #     split='valid',
    #     deterministic=True,
    #     loaded_vocab=tokenizer,
    #     model_emb=model_weight # using the same embedding wight with tranining data
    # )

    data_valid = None

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")

    num_frames = 200
    tubevit = TubeViTLightningModule(
        num_classes=2,
        video_shape = [num_frames, 720, 720, 3],
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        lr=args.lr,
        weight_decay=0.001,
        weight_path="init_tubevit.pt",
        max_epochs=10
    ).to("cuda:1")

    tubevit_reducer = nn.Linear(43744, 4096).to("cuda:1")
    # load tubevit reducer if tubevit_reducer.pt exists
    if os.path.exists("tubevit_reducer.pt"):
        tubevit_reducer.load_state_dict(torch.load("tubevit_reducer.pt"))

    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

    model_diffusion_args = args_to_dict(args, load_defaults_config().keys())

    model, diffusion = create_model_and_diffusion(
        **model_diffusion_args
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to("cuda:0") #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        tubevit=tubevit,
        tubevit_reducer=tubevit_reducer
    ).run_loop()

if __name__ == "__main__":
    main()
