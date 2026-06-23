"""Automated script implementing macro-batch warm-starting (B=32) 
before mutating to an active CENSOR evaluation track at Batch Size 1 on CIFAR-100.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torchvision
import torch.nn as nn
import yaml
import numpy as np
import datetime
import logging
import defense
import inversefed

def init_logger(output_dir):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(output_dir, "hybrid_bs_study.log"), encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(sh)
    return root_logger

if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    with open("./configs_gan_free.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = os.path.join(config['output_dir'], f"hybrid_study_{current_time}")
    os.makedirs(save_dir, exist_ok=True)
    logger = init_logger(save_dir)

    setup = dict(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), dtype=torch.float)
    
    # ----------------------------------------------------------------=====
    # PHASE 1: HIGH-STABILITY MACRO-BATCH INTEL ACCUMULATION (B=32)
    # ----------------------------------------------------------------=====
    print("\n" + "="*80)
    print(" >>> PHASE 1: Launching High-Stability Dataloader (Batch Size = 32)...")
    print("="*80)
    
    defs_warm = inversefed.training_strategy('conservative')
    defs_warm.epochs = 5
    defs_warm.lr = 0.01  
    defs_warm.batch_size = 32

    loss_fn, trainloader_32, validloader_32 = inversefed.construct_dataloaders(
        config['dataset'], defs_warm, data_path=config['data_path']
    )

    # Force strict CIFAR-100 dimension limits
    num_classes = 100
    model, _ = inversefed.construct_model(config['model'], num_classes=num_classes, num_channels=3, seed=config['set_seed'])
    model.to(**setup)

    optimizer = torch.optim.SGD(model.parameters(), lr=defs_warm.lr, momentum=0.9, weight_decay=defs_warm.weight_decay)
    
    warm_start_complete = False
    calibration_epoch = 0
    
    while not warm_start_complete:
        model.train()
        for batch_idx, (inputs_tr, targets_tr) in enumerate(trainloader_32):
            # Scaled up to 1500 to allow the 100-class space adequate optimization time
            if batch_idx >= 1500: break 
            inputs_tr, targets_tr = inputs_tr.to(**setup), targets_tr.to(**setup).long()
            optimizer.zero_grad()
            out_tr = model(inputs_tr)
            l_tr, _, _ = loss_fn(out_tr, targets_tr)
            l_tr.backward()
            optimizer.step()
            
        # Check validation pool over a strict balanced subset slice
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for idx, (inputs_v, targets_v) in enumerate(validloader_32):
                if idx >= 64: break # 64 * 32 = 2048 high-accuracy profile points
                inputs_v, targets_v = inputs_v.to(**setup), targets_v.to(**setup).long()
                val_preds = model(inputs_v).argmax(dim=1)
                val_correct += (val_preds == targets_v).sum().item()
                val_total += targets_v.size(0)
                
        current_acc = (val_correct / val_total) * 100
        calibration_epoch += 1
        print(f" >>> [Warm-up Epoch {calibration_epoch:02d}] Global Validation Accuracy: {current_acc:.2f}%")
        
        # Target calibration accuracy set to a realistic baseline for CIFAR-100 from scratch
        if current_acc >= 50.0:
            print("\n >>> [SUCCESS] 50% CIFAR-100 target reached. Freezing network weights.")
            print(" >>> Shutting down Batch Size 32 dataloaders completely.\n")
            warm_start_complete = True

    # ----------------------------------------------------------------=====
    # PHASE 2: CONVERT ENVIRONMENT TO CENSOR STRATEGY (B=1)
    # ----------------------------------------------------------------=====
    print("="*80)
    print(" >>> PHASE 2: Spawning Evaluation Track Loader Slices (Batch Size = 1)...")
    print("="*80)
    
    defs_eval = inversefed.training_strategy('conservative')
    defs_eval.epochs = config['train_epochs']
    defs_eval.lr = 0.001  
    defs_eval.batch_size = 1

    _, trainloader_1, validloader_1 = inversefed.construct_dataloaders(
        config['dataset'], defs_eval, data_path=config['data_path']
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=defs_eval.lr, momentum=0.9, weight_decay=defs_eval.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=defs_eval.epochs)
    
    # Mount local CIFAR-100 label mapping vector
    try:
        cifar10_classes = trainloader_1.dataset.classes
    except Exception:
        cifar10_classes = [f"Class_{i}" for i in range(100)]

    global_total_attacks = 0  
    global_successful_leaks = 0
    epoch_metrics_log = []

    print("\n" + "="*105)
    print("                CENSOR ACTIVE PIPELINE: BATCH SIZE 1 PRIVACY DECAY EXPERIMENT                ")
    print("="*105)

    eval_iter = iter(trainloader_1)

    for epoch in range(defs_eval.epochs):
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for idx, (inputs_v, targets_v) in enumerate(validloader_1):
                # FIXED: Shifted from 200 to 2000 to match strict class distribution scales
                if idx >= 2000: break  
                inputs_v, targets_v = inputs_v.to(**setup), targets_v.to(**setup).long()
                val_preds = model(inputs_v).argmax(dim=1)
                val_correct += (val_preds == targets_v).sum().item()
                val_total += targets_v.size(0)
        epoch_val_acc = (val_correct / val_total) * 100

        print(f"\n[EPOCH {epoch:02d}/{defs_eval.epochs:02d}] Global Model Validation Base: {epoch_val_acc:.2f}%")
        print("-"*105)
        print(f"{'Test Node':<10} | {'True Class Label':<20} | {'Model Prediction':<20} | {'Attack Guess':<20} | {'Privacy State':<10}")
        print("-"*105)

        epoch_attacks = 0
        epoch_leaks = 0

        for step in range(10):
            try:
                inputs, targets = next(eval_iter)
            except StopIteration:
                eval_iter = iter(trainloader_1)
                inputs, targets = next(eval_iter)
                
            inputs, targets = inputs.to(**setup), targets.to(**setup).long()

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                pred_idx = outputs.argmax(dim=1).item()

            model.train()
            optimizer.zero_grad()
            loss_outputs = model(inputs)
            loss, _, _ = loss_fn(loss_outputs, targets)
            loss.backward()
            raw_gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            optimizer.zero_grad()

            d_param = config['defense_setting']['orthogonal']
            protected_gradients, _ = defense.orthogonal_gradient(
                raw_gradients, model, inputs, targets, 
                trials=config['our_num_tries'], epsilon=d_param, best_loss=loss
            )

            target_fc_grad = None
            for g_tensor in protected_gradients:
                if g_tensor is not None and len(g_tensor.shape) == 2 and g_tensor.shape[0] == num_classes:
                    target_fc_grad = g_tensor.clone().detach()
                    break

            if target_fc_grad is not None:
                row_norms = torch.norm(target_fc_grad, dim=1)
                inferred_label_idx = row_norms.argmax().item()
                
                label_item = targets.item()
                is_leaked = inferred_label_idx == label_item
                
                if is_leaked:
                    epoch_leaks += 1
                    global_successful_leaks += 1
                epoch_attacks += 1
                global_total_attacks += 1  

                true_str = cifar10_classes[label_item]
                pred_str = cifar10_classes[pred_idx]
                guess_str = cifar10_classes[inferred_label_idx]
                state_str = "🚨 LEAKED" if is_leaked else "🛡️ SHIELDED"

                print(f"Sample {step:02d}  | {true_str:<20} | {pred_str:<20} | {guess_str:<20} | {state_str:<10}")

            model.train()
            for sub_step in range(50):
                try:
                    inputs_tr, targets_tr = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(trainloader_1)
                    inputs_tr, targets_tr = next(eval_iter)
                    
                inputs_tr, targets_tr = inputs_tr.to(**setup), targets_tr.to(**setup).long()
                optimizer.zero_grad()
                out_tr = model(inputs_tr)
                l_tr, _, _ = loss_fn(out_tr, targets_tr)
                l_tr.backward()
                optimizer.step()

        scheduler.step()
        epoch_leak_pct = (epoch_leaks / epoch_attacks) * 100 if epoch_attacks > 0 else 0.0
        epoch_metrics_log.append((epoch, epoch_val_acc, epoch_leak_pct))
        print("-"*105)
        print(f" >>> EPOCH SUMMARY: Model Accuracy = {epoch_val_acc:.2f}% | Attack Leakage Rate = {epoch_leak_pct:.2f}%")
        print("="*105)

    # --- FINAL PROGRESSION PROFILE SUMMARY ---
    print("\n" + "#"*70)
    print("               FINAL RESEARCH EXPERIMENT METRICS REPORT              ")
    print("#"*70)
    print(f" Target Network Profile Architecture : {config['model']}")
    print(f" Batch Slicing Configuration        : Size = 1 (Isolated Nodes)")
    print(f" Defense System Core Mechanism      : CENSOR Subspace Projections")
    print("-"*70)
    print(f"{'Epoch Index':<12} | {'Global Model Accuracy':<25} | {'Label Inference Leak Rate':<25}")
    print("-"*70)
    for ep, acc, lk in epoch_metrics_log:
        print(f"Epoch {ep:02d}       | {acc:05.2f}%                    | {lk:05.2f}%")
    print("-"*70)
    final_leak_total = (global_successful_leaks / global_total_attacks) * 100
    print(f" >>> Aggregated Global Experiment Leakage Rate: {final_leak_total:.2f}%")
    print("#"*70)