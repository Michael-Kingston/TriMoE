# trainer
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import tqdm as accelerate_tqdm
from .utils import compute_metrics 
from .utils import FocalLoss 





def train(args, accelerator: Accelerator, model: nn.Module, optimizer, scheduler, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
    """
    Main training loop for the model.

    """
    
    pos_weight = torch.tensor([args.pos_weight], device=accelerator.device) if hasattr(args, 'pos_weight') and args.pos_weight is not None else None
    
    if args.loss_function == 'focal':
        accelerator.print("using FocalLoss")
        criterion = FocalLoss(alpha=0.65, gamma=2.0)
    else:
        accelerator.print(f"using bce pos_weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    
    best_val_auc = 0.0
    epochs_no_improve = 0
    patience = args.patience

    
    training_history = {
        'centroids': [],    
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'val_auc': [],
        'expert_usage': []  
    }

    
    warmup_epochs = 2 
    
    accelerator.print(f"starting training for {args.num_train_epochs} epochs with a patience of {patience}.")
    

    for epoch in range(args.num_train_epochs):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch + 1}/{args.num_train_epochs} ---")

        model.train()
        total_loss_for_epoch = 0.0
        
        
        unwrapped_model = accelerator.unwrap_model(model)
        epoch_expert_usage = [torch.zeros(n_exp, device=accelerator.device) for n_exp in unwrapped_model.moe_layer.num_experts_per_modality]

        
        is_disjoint_moe = hasattr(unwrapped_model, 'moe_layer')
        if epoch < warmup_epochs and is_disjoint_moe:
            unwrapped_model.moe_layer.loss_coef = args.moe_loss_coef * 0.1
            if accelerator.is_main_process: print(f"disjoint MoE aux loss is reduced")
        elif is_disjoint_moe:
            unwrapped_model.moe_layer.loss_coef = args.moe_loss_coef
            if accelerator.is_main_process and epoch == warmup_epochs: print(f"aux loss is  (coef={args.moe_loss_coef}) ---")
        
        progress_bar = accelerate_tqdm(train_loader, desc="Training", disable=not accelerator.is_main_process)
        for step, batch in enumerate(progress_bar):
            batch_data, _ = batch
            if batch_data is None: continue
            
            ts_i, ts_m, ts_t, dem, i_ids, a_m, n_t_m, labels = batch_data

            with accelerator.autocast():
                
                logits, aux_loss, batch_expert_usage = model(*batch_data[:-1], step=step)
                main_loss = criterion(logits, labels)
                
                if args.loss_function == 'focal':
                    total_loss = (main_loss * 25) + aux_loss
                else:
                    total_loss = main_loss + aux_loss

            accelerator.backward(total_loss)
            if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None: scheduler.step()
            optimizer.zero_grad()

            total_loss_for_epoch += total_loss.item()
            
            
            with torch.no_grad():
                for i in range(len(epoch_expert_usage)):
                    epoch_expert_usage[i] += batch_expert_usage[i].to(accelerator.device)
            
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=total_loss.item(), main_loss=main_loss.item(), aux_loss=aux_loss.item())

        avg_train_loss = total_loss_for_epoch / len(train_loader) if len(train_loader) > 0 else 0.0
        accelerator.print(f"avg training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")

        
        val_metrics, _ = evaluate(accelerator, model, val_loader, criterion)
        if test_loader is not None:
            test_metrics, test_loss = evaluate(accelerator, model, test_loader, criterion)
        else:
            test_metrics, test_loss = {}, 0.0
        
        accelerator.wait_for_everyone()

        
        if accelerator.is_main_process:
            
            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['test_loss'].append(test_loss)
            training_history['val_auc'].append(val_metrics.get('auc', 0.0))
            
            centroids_per_modality = [c.detach().cpu().numpy() for c in unwrapped_model.moe_layer.gating_centroids]
            training_history['centroids'].append(centroids_per_modality)
            
            final_epoch_usage = [usage.cpu().numpy() for usage in epoch_expert_usage]
            training_history['expert_usage'].append(final_epoch_usage)

            
            current_auc = val_metrics.get('auc', 0.0)
            print(f"Validation AUROC: {current_auc:.4f} | Best AUROC: {best_val_auc:.4f}")

            if current_auc > best_val_auc:
                best_val_auc = current_auc
                epochs_no_improve = 0
                print(f"new best validation AUROC")
                os.makedirs(args.output_dir, exist_ok=True)
                accelerator.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            else:
                epochs_no_improve += 1
        
        
        if hasattr(unwrapped_model, 'update_dropped_experts'):
            is_within_dropout_period = (epoch + 1) >= args.expert_dropout_epoch_start and (epoch + 1) < args.expert_dropout_epoch_end
            is_final_dropout_epoch = (epoch + 1) == args.expert_dropout_epoch_end

            if is_within_dropout_period:
                accelerator.print(f"updating dropped experts list for upcoming Epoch {epoch + 2}")
                unwrapped_model.update_dropped_experts()
            elif is_final_dropout_epoch:
                accelerator.print(f"expert dropout period ended. Reactivating all experts for subsequent epochs.")
                unwrapped_model.clear_dropped_experts()
        
       
        stop_signal_tensor = torch.tensor([epochs_no_improve], device=accelerator.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_signal_tensor, src=0)
        
        if stop_signal_tensor.item() >= patience:
            accelerator.print(f"early stopping triggered after {patience} epochs")
            break

    accelerator.print("\nTraining loop finished.")
    
    if accelerator.is_main_process:
        history_path = os.path.join(args.output_dir, "training_history")
        os.makedirs(history_path, exist_ok=True)
        print(f"saving training history to {history_path}")

        np.savez(os.path.join(history_path, 'centroids_per_epoch.npz'), 
                 data=np.array(training_history['centroids'], dtype=object))
        np.savez(os.path.join(history_path, 'expert_usage_per_epoch.npz'),
                 data=np.array(training_history['expert_usage'], dtype=object))
        
        metrics_df = pd.DataFrame({
            'epoch': training_history['epoch'],
            'train_loss': training_history['train_loss'],
            'test_loss': training_history['test_loss'],
            'val_auc': training_history['val_auc']
        })
        metrics_df.to_csv(os.path.join(history_path, 'metrics.csv'), index=False)
        print("History saved successfully.")

def evaluate(accelerator: Accelerator, model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    accelerator.print("Starting evaluation...")
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = accelerate_tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process)
        for batch in progress_bar:
            if batch is None:
                continue

            labels = batch[-1]
            
            logits, _, _ = model(*batch[:-1])
            
            
            loss = criterion(logits, labels)
            
            gathered_loss = accelerator.gather(loss.repeat(labels.size(0)))
            total_loss += torch.mean(gathered_loss).item()
            

            gathered_logits, gathered_labels = accelerator.gather_for_metrics((logits, labels))
            all_logits.append(gathered_logits.cpu())
            all_labels.append(gathered_labels.cpu())

    if len(all_logits) == 0:
        accelerator.print("Evaluation found no valid samples.")
       
        return {'auc': 0.0, 'auprc': 0.0, 'f1': 0.0}, 0.0

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = {}
    if accelerator.is_main_process:
        if all_logits.numel() > 0 and all_labels.numel() > 0:
            num_gathered_samples = len(all_labels)
            accelerator.print(f"Evaluation: Gathered {num_gathered_samples} samples for metric calculation.")
            logits_list = all_logits.numpy().flatten().tolist()
            labels_list = all_labels.numpy().flatten().tolist()
            metrics = compute_metrics(logits_list, labels_list)
        else:
            accelerator.print("Warning: No valid samples were gathered during evaluation.")
            metrics = {'auc': 0.0, 'auprc': 0.0, 'f1': 0.0}

    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accelerator.print(f"Evaluation complete. Average Loss: {avg_loss:.4f}")
    
    
    return metrics, avg_loss

