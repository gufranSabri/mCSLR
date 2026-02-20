import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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

from model import SLR_Model
from modules.metrics import wer_list
from modules.parser import get_args_parser
from modules.logger import Logger
from modules.optimizer import build_optimizer, build_scheduler
from modules.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014

def set_rng_state(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def single_loop(model, tokenizer, loader, optimizer, logger, train=True, epoch=None):
    if train: model.train()
    else: model.eval()
    
    total_loss = 0
    evaluation_results = {}
    eval_results = defaultdict(dict)
    
    for i, (src_input) in tqdm(enumerate(loader), total=len(loader), ncols=100, desc=f"{'Training' if train else 'Evaluating'} Epoch {epoch}"):
    # for i, (src_input) in enumerate(loader):
        with torch.set_grad_enabled(model.training):
            output = model(src_input)

        if train:
            optimizer.zero_grad()
            output['total_loss'].backward()
            optimizer.step()
            model.zero_grad()

            loss_value = output['total_loss'].item()
            total_loss += loss_value
            if torch.isnan(output['total_loss']) or torch.isinf(output['total_loss']):
                logger("Loss is {}, stopping training".format(loss_value))
                continue

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
        evaluation_results['wer'] = 200
        for hyp_name in eval_results[name].keys():
            if not 'gls_hyp' in hyp_name: continue

            gls_ref, gls_hyp = [], []
            for name in eval_results:
                dataset = eval_results[name]['dataset']
                ref = eval_results[name]['gls_ref']
                hyp = eval_results[name][hyp_name]

                if dataset.lower() == 'phoenix-2014t':
                    ref = clean_phoenix_2014_trans(ref)
                    hyp = clean_phoenix_2014_trans(hyp)
                elif dataset.lower() == 'phoenix-2014':
                    ref = clean_phoenix_2014(ref)
                    hyp = clean_phoenix_2014(hyp)

                gls_ref.append(ref)
                gls_hyp.append(hyp)

            wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
            evaluation_results[hyp_name.replace('gls_hyp', '') + 'wer_list'] = wer_results
            evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])

    return total_loss / len(loader) if train else evaluation_results['wer']


def main(args, config):
    device = torch.device(args.device)

    logger = Logger(file_path=os.path.join(args.work_dir, "train.log"))
    for key, value in vars(args).items():
        logger(f"{key}: {value}")
    
    train_dataloader = setup_dataloaders(args, config, phase='train')
    dev_dataloader = setup_dataloaders(args, config, phase='dev')
    test_dataloader = setup_dataloaders(args, config, phase='test')
    
    logger("\n")
    logger("Datasets loaded successfully.")
    logger(f"Number of training samples: {len(train_dataloader.dataset)}")
    logger(f"Number of dev samples: {len(dev_dataloader.dataset)}")
    logger(f"Number of test samples: {len(test_dataloader.dataset)}\n")

    model = SLR_Model(cfg=config, args=args).to(device)
    if args.mode == 'test':
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'best_model.pth'), map_location=args.device))

    optimizer = build_optimizer(config=config['training']['optimization'], model=model)
    scheduler, scheduler_type = build_scheduler(config=config['training']['optimization'], optimizer=optimizer)

    logger(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.\n")
    logger(f"Optimizer: {optimizer.__class__.__name__}")
    logger(f"Scheduler: {scheduler.__class__.__name__} (Type: {scheduler_type})\n")

    logger(f"Starting {'training' if args.mode == 'train' else 'testing'}...\n")
    best_wer = 1000
    for epoch in range(args.epochs):
        scheduler.step()
        start_time = time.time()

        # loss = single_loop(
        #     model, model.net.gloss_tokenizer, 
        #     train_dataloader, optimizer, logger, 
        #     train=True, epoch=epoch
        # )

        wer = single_loop(
            model, model.net.gloss_tokenizer, 
            dev_dataloader, optimizer, logger, 
            train=False, epoch=epoch
        )

        logger(f"Epoch [{epoch}/{args.epochs}]")
        logger(f" - Loss: {loss:.4f}")
        logger(f" - WER: {wer:.2f}%")
        
        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(args.work_dir, "best_model.pt"))
            logger(f"New best model saved.")
        
        logger(f"Epoch time: {time.time() - start_time:.2f} seconds")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        setattr(args, key, value)

    if not os.path.exists("work_dir"): os.makedirs("work_dir")
    if args.mode == 'test': args.epochs = 1

    if args.mode == 'train':
        os.makedirs(args.work_dir, exist_ok=True)

    if args.mode == 'train':
        shutil.copy2(args.config, args.work_dir)
        shutil.copy2("./main.py", args.work_dir)
        shutil.copy2("./dataset.py", args.work_dir)
        shutil.copy2("./model.py", args.work_dir)
        shutil.copy2("./configs/combined.yaml", args.work_dir)
    
    set_rng_state(42)
    main(args, config)