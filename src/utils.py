#  utils

import os
import random
import csv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # have to make the model deterministic for reproducibility
    torch.backends.cudnn.deterministic = True
    # there is a chance if this is on that the model won't be deterministic, as may consider different algs each time
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    """Focal Loss implementation for alternative class balancing loss function that gives more weight to difficult examples"""
    """Multiplied by scaling factor in train as loss is too low (<0.1) to work with other param configs (aux loss 0.1)"""
    def __init__(self, alpha: float = 0.65, gamma: float = 2.0):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
         
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        
        return focal_loss.mean()

def save_results_sequentially(args, metrics, filepath):
    """
    Saves a comprehensive list of experiment arguments and results to a CSV file.
    Creates the file and header if it doesn't exist.
    """
    
    header = ['output_dir', 'seed', 'model_architecture', 'use_missing_embeds','fusion_strategy', 'num_experts', 'pretrained_centroids_path',
        
        
        'early_fusion_strategy', 'num_shared_experts', 'moe_input_dim',

        
        'top_k', 'moe_loss_coef', 'moe_noise_eps', 'gating_temp', 'use_stabilized_expert',
        
        
        'learning_rate', 'main_lr', 'bert_lr_factor', 'weight_decay', 'patience',
        'loss_function', 'dropout_p', 'batch_size', 'num_train_epochs',

        
        'ts_time_embedder', 'ts_attention_type', 'use_residual_text_block', 'classifier_head',
        'max_length', 'num_of_notes',
        
        
        'test_auc', 'test_auprc', 'test_f1'
    ]
    

    row = {
       
        'output_dir': args.output_dir,
        'seed': args.seed,
        
        
        'model_architecture': getattr(args, 'model_architecture', 'N/A'),
        'use_missing_embeds': getattr(args, 'use_missing_embeds', False),
        
       
        'fusion_strategy': getattr(args, 'fusion_strategy', 'N/A'),
        'num_experts': str(getattr(args, 'num_experts', 'N/A')), 
        'pretrained_centroids_path': getattr(args, 'pretrained_centroids_path', 'N/A'),
        
       
        'early_fusion_strategy': getattr(args, 'early_fusion_strategy', 'N/A'),
        'num_shared_experts': getattr(args, 'num_shared_experts', 'N/A'),
        'moe_input_dim': getattr(args, 'moe_input_dim', 'N/A'),

        
        'top_k': args.top_k,
        'moe_loss_coef': args.moe_loss_coef,
        'moe_noise_eps': args.moe_noise_eps,
        'gating_temp': args.gating_temp,
        'use_stabilized_expert': getattr(args, 'use_stabilized_expert', False),
        
        'learning_rate': args.learning_rate,
        'main_lr': args.main_lr,
        'bert_lr_factor': args.bert_lr_factor,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'loss_function': getattr(args, 'loss_function', 'N/A'),
        'dropout_p': args.dropout_p,
        'batch_size': args.batch_size,
        'num_train_epochs': args.num_train_epochs,
        
        
        'ts_time_embedder': getattr(args, 'ts_time_embedder', 'N/A'),
        'ts_attention_type': getattr(args, 'ts_attention_type', 'N/A'),
        'use_residual_text_block': getattr(args, 'use_residual_text_block', False),
        'classifier_head': getattr(args, 'classifier_head', 'N/A'),
        'max_length': args.max_length,
        'num_of_notes': args.num_of_notes,

        
        'test_auc': metrics.get('auc', 0.0),
        'test_auprc': metrics.get('auprc', 0.0),
        'test_f1': metrics.get('f1', 0.0)
    }

    try:
        file_exists = os.path.isfile(filepath)
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        print(f"Successfully appended results to {filepath}")

    except Exception as e:
        print(f"An error occurred while saving results to CSV: {e}")        

def analyze_modality_importance(model_state_dict):
    """Prints the learned modality fusion weights from a trained model."""
    print("\n--- Modality Importance Analysis ---")
    
    if 'moe_layer.fusion_weights' in model_state_dict:
        
        raw_weights = model_state_dict['moe_layer.fusion_weights']
        # convert to probabilities using softmax
        softmax_weights = F.softmax(raw_weights, dim=0).cpu().numpy()
        
        modalities = ["Demographics", "Time-Series", "Text"]
        
        print("Learned Modality Weights (Probabilities):")
        for i, modality in enumerate(modalities):
            print(f"- {modality}: {softmax_weights[i]:.4f}")
    else:
        print("No learnable fusion weights found in this model (fusion_strategy was likely 'sum').")

def compute_metrics(logits_list, labels_list):
    """This function turns the model logits into probabilities for binary classification with the sigmoid function, and calculates metrics"""
    # turn true labels and model logits to np arrays for faster calculations
    y_pred_logits, y_true = np.array(logits_list), np.array(labels_list)
    # sigmoid fuctions to turn logits to predictions
    y_prob = 1 / (1 + np.exp(-y_pred_logits))
    #  0.5 threshold for positive or negative prediction
    preds = (y_prob > 0.5).astype(int)
    metrics = {}
    
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        metrics['auprc'] = auc(rec, prec)
        metrics['f1'] = f1_score(y_true, preds, zero_division=0)
    
    except ValueError:
        metrics['auc'], metrics['auprc'], metrics['f1'] = 0.0, 0.0, 0.0
    return metrics