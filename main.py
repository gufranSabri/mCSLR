import os
import shutil
import time
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
from dataset import setup_dataloaders
import torch.optim as optim

from model import SLR_Model
from utils.metrics import wer_list
from utils.parser import get_args_parser
from utils.logger import Logger
from utils.optimizer import build_scheduler
from utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014

def set_rng_state(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def single_loop(model, tokenizer, loader, optimizer, logger, train=True, epoch=None, slurm_mode=True):
    if train: model.train()
    else: model.eval()
    
    total_loss = 0
    evaluation_results = {}
    eval_results = defaultdict(dict)
    
    if not slurm_mode:
        loader = tqdm(loader, total=len(loader), ncols=100, leave=False)

    for i, (src_input) in enumerate(loader):
        optimizer.zero_grad()
        with torch.set_grad_enabled(model.training):
            output = model(src_input)

        if train:
            output['total_loss'].backward()

            # print gradient for each parameter
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         logger(f"Gradient for {name}: {param.grad.norm().item():.4f}")
            #     else:
            #         logger(f"Gradient for {name} is None")

            # exit()

            optimizer.step()

            loss_value = output['total_loss'].item()
            total_loss += loss_value
            if torch.isnan(output['total_loss']) or torch.isinf(output['total_loss']):
                logger("Loss is {}, stopping training".format(loss_value))
                continue

            if not slurm_mode:
                loader.set_postfix_str(f"Loss: {loss_value:.4f}")

        else:
            for k, gls_logits in output.items():
                if not 'gloss_logits' in k: continue
                
                logits_name = k.replace('gloss_logits', '')
                ctc_decode_output = model.net.decode(
                    gloss_logits=gls_logits, beam_size=5,
                    input_lengths=output['input_lengths']
                )
                batch_pred_gls = tokenizer.convert_ids_to_tokens(ctc_decode_output)

                for name, gls_hyp, gls_ref, dataset in zip(src_input['name'], batch_pred_gls, src_input['gloss'], src_input['datasets']):
                    eval_results[name][f'{logits_name}gls_hyp'] = ' '.join(gls_hyp)
                    eval_results[name]['gls_ref'] = gls_ref
                    eval_results[name]['dataset'] = dataset

    if not train:
        pred_logger = Logger(file_path=os.path.join(args.work_dir, "preds", f"predictions_epoch_{epoch}.log"))
        for hyp_name in eval_results[name].keys():
            if not 'gls_hyp' in hyp_name: continue

            pred_logger(f"\n=== Evaluating {hyp_name} ===\n", console_print=False)

            gls_ref, gls_hyp = {dataset:[] for dataset in args.datasets}, {dataset:[] for dataset in args.datasets}
            for name in eval_results:
                dataset = eval_results[name]['dataset']
                ref = eval_results[name]['gls_ref']
                hyp = eval_results[name][hyp_name]

                if dataset.lower() == 'phoenix2014-t':
                    ref = clean_phoenix_2014_trans(ref).lower()
                    hyp = clean_phoenix_2014_trans(hyp).lower()
                elif dataset.lower() == 'phoenix-2014':
                    ref = clean_phoenix_2014(ref).lower()
                    hyp = clean_phoenix_2014(hyp).lower()

                gls_ref[dataset].append(ref)
                gls_hyp[dataset].append(hyp)

                pred_logger(f"Ref: {ref}\nHyp: {hyp}\n", console_print=False)

            for dataset in args.datasets:
                wer_results = wer_list(hypotheses=gls_hyp[dataset], references=gls_ref[dataset])
                evaluation_results[hyp_name.replace('gls_hyp', '') + f'{dataset}_wer_list'] = wer_results
                evaluation_results[dataset] = min(wer_results['wer'], evaluation_results.get(dataset, 1000))

            pred_logger(f"\n==============================\n", console_print=False)

    return total_loss / len(loader) if train else evaluation_results


def main(args, config):
    device = torch.device(args.device)
    checkpoint_path = os.path.join(args.work_dir, "checkpoint_last.pt")

    logger = Logger(file_path=os.path.join(args.work_dir, "train.log"))

    if not args.resume_training:
        for key, value in vars(args).items():
            logger(f"{key}: {value}")
    
    train_dataloader = setup_dataloaders(args, config, phase='train')
    dev_dataloader = setup_dataloaders(args, config, phase='dev')
    test_dataloader = setup_dataloaders(args, config, phase='test')
    
    if not args.resume_training:
        logger("\n")
        logger("Datasets loaded successfully.")
        logger(f"Number of training samples: {len(train_dataloader.dataset)}")
        logger(f"Number of dev samples: {len(dev_dataloader.dataset)}")
        logger(f"Number of test samples: {len(test_dataloader.dataset)}\n")

    model = SLR_Model(cfg=config, args=args).to(device)
    if args.mode == 'test':
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'best_model.pt'), map_location=args.device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        eps=1e-8, 
        weight_decay=0.01, 
        betas=(0.9, 0.98)
    )
    # scheduler, scheduler_type = build_scheduler(config=config['training']['optimization']['scheduler'], optimizer=optimizer)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['training']['optimization']['scheduler']['step'], 
        gamma=config['training']['optimization']['scheduler']['factor']
    )
    scheduler_type = "epoch"

    if not args.resume_training:
        logger(f"Model initialized with {sum(p.numel() for p in model.parameters())/1000000:.2f} million parameters.\n")
        logger(f"Optimizer: {optimizer.__class__.__name__}")
        logger(f"Scheduler: {scheduler.__class__.__name__} (Type: {scheduler_type})\n")

    if not args.resume_training: logger(f"Starting {'training' if args.mode == 'train' else 'testing'}...\n")
    
    best_wer = 1000
    start_epoch = 0
    patience = config['training']['optimization'].get('patience', 16)
    if args.mode == 'train' and args.resume_training:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Resume requested because work directory exists, but checkpoint not found at: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'patience' in checkpoint:
            patience = checkpoint['patience']

        best_wer = checkpoint.get('best_wer', best_wer)
        start_epoch = checkpoint.get('epoch', -1) + 1
        logger(f"Resuming training from checkpoint: {checkpoint_path}")
        logger(f"Resumed at epoch {start_epoch} with best WER {best_wer:.2f}%\n")


    for epoch in range(start_epoch, args.epochs):
        if scheduler_type == "epoch": scheduler.step()
        if scheduler_type == "validation": scheduler.step(best_wer)
        logger(f"Epoch [{epoch}/{args.epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")

        start_time = time.time()

        if args.mode == 'train':
            loss = single_loop(
                model, model.net.gloss_tokenizer, 
                train_dataloader, optimizer, logger, 
                train=True, epoch=epoch, slurm_mode=args.slurm_mode
            )

        wers = single_loop(
            model, model.net.gloss_tokenizer, 
            dev_dataloader, optimizer, logger, 
            train=False, epoch=epoch, slurm_mode=args.slurm_mode
        )
        avg_wer = sum(wers[dataset] for dataset in args.datasets) / len(args.datasets)

        
        logger(f" - Loss: {loss:.4f}")
        for dataset in args.datasets:
            logger(f" - {dataset} WER: {wers[dataset]:.2f}%")
        logger(f" - Average WER: {avg_wer:.2f}%")
        
        if avg_wer < best_wer:
            best_wer = avg_wer
            torch.save(model.state_dict(), os.path.join(args.work_dir, "best_model.pt"))
            
            patience = config['training']['optimization'].get('patience', 16)
            logger(f"New best model saved.")
        else:
            patience -= 1
            logger(f"No improvement in WER. Patience reduced to {patience}.")

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_wer': best_wer,
                'patience': patience
            },
            checkpoint_path,
        )
        
        logger(f"Epoch time: {((time.time() - start_time)/60):.2f} minutes\n\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        setattr(args, key, value)

    if not os.path.exists("work_dir"): os.makedirs("work_dir")
    if args.mode == 'test': args.epochs = 1

    resume_training = args.mode == 'train' and os.path.exists(args.work_dir)
    args.resume_training = resume_training

    if args.mode == 'train':
        os.makedirs(args.work_dir, exist_ok=True)
        os.makedirs(os.path.join(args.work_dir, "preds"), exist_ok=True)

    if args.mode == 'train' and not resume_training:
        shutil.copy2(args.config, args.work_dir)
        shutil.copy2("./main.py", args.work_dir)
        shutil.copy2("./dataset.py", args.work_dir)
        shutil.copy2("./model.py", args.work_dir)
        shutil.copy2("./configs/baseline.yaml", args.work_dir)
    
    set_rng_state(42)
    main(args, config)