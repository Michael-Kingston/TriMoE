# run_training.py

import os

import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


from src.data import TSNote_Irg, train_collate_wrapper, eval_collate_wrapper
from src.moe import TriMoE, EarlyFusionMoEModel 
from src.trainer import train, evaluate
from src.utils import set_seed, save_results_sequentially, analyze_modality_importance, FocalLoss


def create_optimizer_with_groups(model, args):
    """AdamW optimiser."""
    # lists to hold params for each group
    bert_params = []
    time_embedder_params = []
    centroid_params = []
    other_params = []

    # sort params into groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'gating_centroids' in name:
            centroid_params.append(param)
        elif 'time_embedder' in name:
            time_embedder_params.append(param)
        elif 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
            
    
    time_embedder_lr = args.main_lr * 0.1 
    bert_lr = args.learning_rate * args.bert_lr_factor
    centroid_lr = args.main_lr * 20

    optimizer_groups = [
        {'params': bert_params, 'lr': bert_lr, 'name': 'BERT'},
        {'params': time_embedder_params, 'lr': time_embedder_lr, 'name': 'TimeEmbedder'},
        {'params': centroid_params, 'lr': centroid_lr, 'name': 'Centroids'},
        {'params': other_params, 'lr': args.main_lr, 'name': 'Other'}
    ]
    
    
    active_optimizer_groups = [g for g in optimizer_groups if g['params']]

    print("optimizer groups")
    for i, group in enumerate(active_optimizer_groups):
        
        print(f"group {i} ({group['name']}): {len(group['params'])} params, LR: {group['lr']:.2e}")
    
    
    return torch.optim.AdamW(active_optimizer_groups, weight_decay=args.weight_decay)


def main():
    parser = argparse.ArgumentParser(description="TriMOE Training with Accelerate (rewritten, robust)")
    
    # data args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./saved_models/exp_default")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--results_file', type=str, default=None, help="Path to a shared CSV file to append final test results.")

    # train args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Base learning rate for AdamW (often used for BERT).")
    parser.add_argument('--main_lr', type=float, default=1e-3, help="Learning rate for the non-BERT parts of the model.")
    parser.add_argument('--bert_lr_factor', type=float, default=0.2, help="Multiplier for the learning_rate for BERT params.")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--loss_function', type=str, default='bce', choices=['bce', 'focal'], help="Loss function to use ('bce' or 'focal').")

    # model args
    parser.add_argument('--model_architecture', type=str, default='disjoint_moe', choices=['disjoint_moe', 'early_fusion_moe'], help="Choose the main model architecture to run.")
    parser.add_argument('--use_missing_embeds', action='store_true', help="If set, use learnable embeddings for missing modalities.")
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--fusion_strategy', type=str, default='sum', choices=['sum', 'weighted'], help="For disjoint_moe: How to fuse modality outputs.")
    parser.add_argument('--classifier_head', type=str, default='simple', choices=['simple', 'advanced'], help="For disjoint_moe: Type of classifier head.")

    # encoder args
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_of_notes', type=int, default=10)
    parser.add_argument('--text_transformer_layers', type=int, default=2)
    parser.add_argument('--use_residual_text_block', action='store_true', help="If set, adds a residual block to refine the final text embedding.")
    parser.add_argument('--dem_embed_dim', type=int, default=32)
    parser.add_argument('--ts_embed_dim', type=int, default=128)
    parser.add_argument('--text_embed_dim', type=int, default=256)
    parser.add_argument('--tt_max', type=int, default=48)
    parser.add_argument('--ts_time_embedder', type=str, default='simple', choices=['simple', 'time2vec'], help="Type of time embedder for the time-series encoder.")
    parser.add_argument('--ts_attention_type', type=str, default='standard', choices=['standard', 'irregular_aware'], help="Type of attention for the time-series encoder.")
    parser.add_argument('--lambda_sparse', type=float, default=1e-3, help="Sparsity loss coefficient for TabNet.")

    # moe args
    parser.add_argument('--num_experts', type=int, nargs='+', default=[4, 8, 8], help="For disjoint_moe: Number of experts for each modality.")
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--moe_output_dim', type=int, default=512)
    parser.add_argument('--moe_hidden_dim', type=int, default=1024)
    parser.add_argument('--moe_loss_coef', type=float, default=0.01)
    parser.add_argument('--moe_noise_eps', type=float, default=0.0, help="Noise added to gating logits for exploration.")
    parser.add_argument('--gating_temp', type=float, default=1.0)
    parser.add_argument('--use_stabilized_expert', action='store_true', help="If set, use LayerNorm/GELU experts instead of ReLU experts.")
    parser.add_argument('--pretrained_centroids_path', type=str, default=None, help="For disjoint_moe: Path to pre-trained gating centroids.")
    
    # complete expert dropout args
    parser.add_argument('--expert_dropout_min_k', type=int, default=0, 
                        help="The minimum number of experts to drop each epoch (inclusive). Set to 0 to disable.")
    parser.add_argument('--expert_dropout_max_k', type=int, default=0,
                        help="The maximum number of experts to drop each epoch (inclusive).")
    parser.add_argument('--expert_dropout_persistence_prob', type=float, default=0.0,
                        help="The probability (0.0 to 1.0) of keeping the most-used expert dropped for a second epoch.")
    parser.add_argument('--expert_dropout_epoch_start', type=int, default=0, 
                        help="The first epoch number to apply complete expert dropout (1-based).")
    parser.add_argument('--expert_dropout_epoch_end', type=int, default=999, 
                        help="The final epoch number to apply complete expert dropout (inclusive).")
    parser.add_argument('--n_experts_to_drop', type=int, default=0, 
                    help="For DisjointMoE: The fixed number of most-utilized experts to drop each epoch.")
    
    # early fusion args 
    parser.add_argument('--moe_input_dim', type=int, default=512, help="For early_fusion_moe: The unified input dimension for the shared MoE layer.")
    parser.add_argument('--num_shared_experts', type=int, default=16, help="For early_fusion_moe: Number of experts in the shared MoE layer.")
    parser.add_argument('--early_fusion_strategy', type=str, default='concat', choices=['concat', 'weighted'], help="For early_fusion_moe")
    
    args = parser.parse_args()
    set_seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    accelerator.print("--- Experiment Arguments ---")
    for k, v in vars(args).items(): accelerator.print(f"{k}: {v}")
    accelerator.print("--------------------------")

    args.tabnet_params = {
        "n_d": 12, 
        "n_a": 12, 
        "n_steps": 3, 
        "gamma": 1.5, 
        "momentum": 0.02, 
        "mask_type": "sparsemax",
        "virtual_batch_size": 8 
    }
    
    accelerator.print("--- Loading base models ---")
    BioBert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    accelerator.print("\n--- Preparing DataLoaders ---")
    train_dataset = TSNote_Irg(args, 'train', tokenizer)
    val_dataset = TSNote_Irg(args, 'val', tokenizer)
    test_dataset = TSNote_Irg(args, 'test', tokenizer)

    if not train_dataset.data:
        accelerator.print("Training dataset not loaded. Exiting.")
        return
        
    labels = [int(d['label']) for d in train_dataset.data if 'label' in d]
    pos = sum(labels)
    neg = len(labels) - pos
    args.pos_weight = (neg / (pos + 1e-12)) if pos > 0 else 1.0
    accelerator.print(f"Computed pos_weight for BCE: {args.pos_weight:.4f} (pos={pos}/neg={neg})")
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_collate_wrapper, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=eval_collate_wrapper, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=eval_collate_wrapper, num_workers=4, pin_memory=True) if test_dataset.data else None

    if not train_loader or not val_loader:
        accelerator.print("data loader error")
        return
        
    args.dem_input_dim = train_dataset.dem_dim
    args.ts_input_dim = train_dataset.ts_dim
    accelerator.print(f"demographics input dim: {args.dem_input_dim}")
    accelerator.print(f"time series input dim: {args.ts_input_dim}")
    
    accelerator.print(f"\n model: {args.model_architecture} ")

    if args.model_architecture == 'disjoint_moe':
        pretrained_centroids_list = None
        if args.pretrained_centroids_path:
            if os.path.exists(args.pretrained_centroids_path):
                accelerator.print(f"pretrained centroids from {args.pretrained_centroids_path}...")
                centroids_npz = np.load(args.pretrained_centroids_path)
                pretrained_centroids_list = [
                    torch.from_numpy(centroids_npz['dem']),
                    torch.from_numpy(centroids_npz['ts']),
                    torch.from_numpy(centroids_npz['text'])
                ]
            else:
                accelerator.print(f"pretrained centroids path not found {args.pretrained_centroids_path}.")
        model = TriMoE(config=args, biobert_model=BioBert, pretrained_centroids=pretrained_centroids_list)
    
    elif args.model_architecture == 'early_fusion_moe':
        model = EarlyFusionMoEModel(config=args, biobert_model=BioBert)
        
    else:
        raise ValueError(f"unknown model architecture {args.model_architecture}")

    accelerator.print("parameter groups for differential learning rates setting up")
    for name, param in model.text_encoder.bert.named_parameters():
        if 'encoder.layer.11' not in name: param.requires_grad = False
        else: accelerator.print(f" > Unfrozen for fine-tuning: {name}")

    optimizer = create_optimizer_with_groups(model, args)

    accelerator.print("preparing model, optimiser, and dataloaders with accelerate")
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    num_training_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_training_steps
    )

    train(args, accelerator, model, optimizer, scheduler, train_loader, val_loader, test_loader)

    accelerator.wait_for_everyone()

    
    final_metrics = None
    state_dict = None

    if test_loader is not None:
        accelerator.print("\n evaluating best model on the test set")
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        if accelerator.is_main_process and not os.path.exists(best_model_path):
            print(f"best model not found at {best_model_path}. skipping final test evaluation")
        else:
            accelerator.wait_for_everyone()
            
            unwrapped_model = accelerator.unwrap_model(model)
            
            if hasattr(unwrapped_model, 'clear_dropped_experts'):
                accelerator.print("clearing any dropped experts before final test evaluation.")
                unwrapped_model.clear_dropped_experts()

            accelerator.print(f"loading best model weights from {best_model_path} for final testing")
            state_dict = torch.load(best_model_path, map_location=accelerator.device)
            unwrapped_model.load_state_dict(state_dict)
                      
            pos_weight_tensor = torch.tensor([args.pos_weight], device=accelerator.device)
            final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if args.loss_function == 'bce' else FocalLoss(alpha=0.65, gamma=2.0)
                        
            final_metrics, _ = evaluate(accelerator, model, test_loader, final_criterion)

            if accelerator.is_main_process:
                print(f"\n final for {args.output_dir} ---")
                for k, v in final_metrics.items(): print(f"Test {k.upper()}: {v:.4f}")
            
            if accelerator.is_main_process and args.model_architecture == 'disjoint_moe' and args.fusion_strategy == 'weighted' and state_dict is not None:
                analyze_modality_importance(state_dict)
    else:
        accelerator.print("test data not found or loader is not prepared")
    
    if accelerator.is_main_process and args.results_file and final_metrics is not None:
        accelerator.print(f"saving results to file {args.results_file}")
        save_results_sequentially(args, final_metrics, args.results_file)

if __name__ == '__main__':
    main()