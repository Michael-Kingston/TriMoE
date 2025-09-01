# model
import os
import random
from typing import List
import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F
from typing import List


from .model import Expert, StabilizedExpert, SparseDispatcher, DisjointSparseMoE
from .encoders import DemographicsTabNetEncoder, TimeSeriesEncoder, TimeAwareTextEncoder
from .encoders import DemographicsTabNetEncoder, TimeSeriesEncoder, TimeAwareTextEncoder



class EarlyFusionMoEModel(nn.Module):
    def __init__(self, config, biobert_model):
        super().__init__()
        self.config = config
        self.noise_eps = getattr(config, 'moe_noise_eps', 0.0)
        self.gating_temp = getattr(config, 'gating_temp', 1.0)
        self.early_fusion_strategy = getattr(config, 'early_fusion_strategy', 'concat')

        
        self.dem_encoder = DemographicsTabNetEncoder(
            config.dem_input_dim,
            config.dem_embed_dim,
            config.tabnet_params
        )
        self.ts_encoder = TimeSeriesEncoder(
            input_dim=config.ts_input_dim,
            embed_dim=config.ts_embed_dim,
            embed_time=16,
            num_heads=2,
            attention_type=getattr(config, 'ts_attention_type', 'standard'),
            time_embedder_type=getattr(config, 'ts_time_embedder', 'simple'),
            tt_max=config.tt_max
        )
        self.text_encoder = TimeAwareTextEncoder(
            biobert_model,
            config.text_embed_dim,
            transformer_heads=8,
            transformer_layers=config.text_transformer_layers,
            use_residual_block=getattr(config, 'use_residual_text_block', False)
        )

        
        self.use_missing_embeds = getattr(config, 'use_missing_embeds', False)
        if self.use_missing_embeds:
            print("INFO: [EarlyFusionMoE] Using learnable embeddings for missing modalities.")
            self.missing_dem_embedding = nn.Parameter(torch.empty(config.dem_embed_dim))
            self.missing_ts_embedding = nn.Parameter(torch.empty(config.ts_embed_dim))
            self.missing_text_embedding = nn.Parameter(torch.empty(config.text_embed_dim))
            
            is_stabilized = getattr(config, 'use_stabilized_expert', False)
            with torch.no_grad():
                self._initialize_embedding(self.missing_dem_embedding, is_stabilized)
                self._initialize_embedding(self.missing_ts_embedding, is_stabilized)
                self._initialize_embedding(self.missing_text_embedding, is_stabilized)

        
        if self.early_fusion_strategy == 'weighted':
            print("INFO: [EarlyFusionMoE] Using LEARNED WEIGHTED fusion.")
            fusion_embed_dim = getattr(config, 'fusion_embed_dim', 256)
            self.dem_fusion_proj = nn.Linear(config.dem_embed_dim, fusion_embed_dim)
            self.ts_fusion_proj = nn.Linear(config.ts_embed_dim, fusion_embed_dim)
            self.text_fusion_proj = nn.Linear(config.text_embed_dim, fusion_embed_dim)
            self.fusion_weights = nn.Parameter(torch.ones(3))
            fused_embedding_dim = fusion_embed_dim
        else: 
            print("INFO: [EarlyFusionMoE] Using CONCATENATION for fusion.")
            fused_embedding_dim = config.dem_embed_dim + config.ts_embed_dim + config.text_embed_dim
        
        moe_input_dim = getattr(config, 'moe_input_dim', 512)
        self.fusion_projection = nn.Linear(fused_embedding_dim, moe_input_dim)
        self.fusion_layernorm = nn.LayerNorm(moe_input_dim)
        
        
        num_shared_experts = getattr(config, 'num_shared_experts', 16)
        self.k = config.top_k
        print("INFO: [EarlyFusionMoE] Using CENTROID-BASED gating.")
        self.gating_centroids = nn.Parameter(torch.randn(num_shared_experts, moe_input_dim) * 0.4)
        
        ExpertChoice = StabilizedExpert if getattr(config, 'use_stabilized_expert', False) else Expert
        self.experts = nn.ModuleList([
            ExpertChoice(moe_input_dim, config.moe_output_dim, config.moe_hidden_dim, config.dropout_p)
            for _ in range(num_shared_experts)
        ])
        
        
        self.expert_dropout_min_k = getattr(config, 'expert_dropout_min_k', 0)
        self.expert_dropout_max_k = getattr(config, 'expert_dropout_max_k', 0)
        self.expert_dropout_persistence_prob = getattr(config, 'expert_dropout_persistence_prob', 0.0)
        
        self.is_dropout_enabled = self.expert_dropout_min_k > 0 and self.expert_dropout_max_k > 0

        if self.is_dropout_enabled:
            print(f"INFO: [EarlyFusionMoE] Initializing stochastic expert dropout.")
            print(f"    - Dropping experts between {self.expert_dropout_min_k} and {self.expert_dropout_max_k}.")
            print(f"    - Persistence probability for top expert: {self.expert_dropout_persistence_prob:.2f}")
            
            self.register_buffer('expert_load_tracker', torch.zeros(num_shared_experts))
            self.dropped_expert_indices = []
            self.expert_to_persist_drop = None 
        
        
        self.final_layernorm = nn.LayerNorm(config.moe_output_dim)
        self.classifier_head = getattr(config, 'classifier_head', 'simple')
        
        if self.classifier_head == 'advanced':
            print("INFO: [EarlyFusionMoE] Using a non-linear classification head.")
            self.proj1 = nn.Linear(config.moe_output_dim, config.moe_output_dim)
            self.proj2 = nn.Linear(config.moe_output_dim, config.moe_output_dim)
            self.head_dropout = nn.Dropout(config.dropout_p)
            self.out_layer = nn.Linear(config.moe_output_dim, 1)
        else: # 'simple'
            print("INFO: [EarlyFusionMoE] Using a simple linear classification head.")
            self.classifier = nn.Linear(config.moe_output_dim, 1)
            
        self.loss_coef = config.moe_loss_coef

    def _initialize_embedding(self, embedding_tensor, is_stabilized):
        std = 0.4 if is_stabilized else 0.6
        nn.init.normal_(embedding_tensor, mean=0.0, std=std)

    def _compute_moe_loss(self, gates):
        load = (gates > 0).float().sum(0)
        importance = gates.sum(0)
        eps = 1e-10
        cv_squared_load = load.var() / (load.mean()**2 + eps) if load.mean() > eps else 0.0
        cv_squared_importance = importance.var() / (importance.mean()**2 + eps) if importance.mean() > eps else 0.0
        return cv_squared_load + cv_squared_importance

    def update_dropped_experts(self):
        """
        Implements a simpler, more correct stochastic and persistent dropout.
        
        """
        if not self.is_dropout_enabled:
            return

        if self.expert_load_tracker.sum() == 0:
            self.dropped_expert_indices = []
            return
            
        most_used_expert_this_epoch = torch.argmax(self.expert_load_tracker).item()

        new_dropped_set = set()
        new_dropped_set.add(most_used_expert_this_epoch)

        if self.expert_to_persist_drop is not None:
            new_dropped_set.add(self.expert_to_persist_drop)
            print(f"    [Expert Dropout Persistence] Expert ({self.expert_to_persist_drop}) is kept dropped for a second epoch.")

        num_to_drop = random.randint(self.expert_dropout_min_k, self.expert_dropout_max_k)
        
        available_experts = [i for i in range(len(self.expert_load_tracker)) if i not in new_dropped_set]
        
        if len(new_dropped_set) < num_to_drop and available_experts:
            num_to_add = num_to_drop - len(new_dropped_set)
            randomly_chosen = random.sample(available_experts, min(num_to_add, len(available_experts)))
            new_dropped_set.update(randomly_chosen)

        self.dropped_expert_indices = list(new_dropped_set)

        if random.random() < self.expert_dropout_persistence_prob:
            self.expert_to_persist_drop = most_used_expert_this_epoch
        else:
            self.expert_to_persist_drop = None
            
        print(f"  [Expert Dropout] Final dropped list for next epoch: {self.dropped_expert_indices}")
        
        self.expert_load_tracker.zero_()

    def clear_dropped_experts(self):
        """Resets all dropout state, reactivating all experts."""
        if self.is_dropout_enabled:
            print("  [Expert Dropout] Clearing all dropped experts. All experts are now active.")
            self.dropped_expert_indices = []
            self.expert_to_persist_drop = None

    def forward(self, ts_input, ts_mask, ts_tt, dem, input_ids, attn_mask, note_time_mask, step=None):
        dem_embedding, M_loss = self.dem_encoder(dem)
        ts_embedding = self.ts_encoder(ts_input, ts_tt, ts_mask)
        text_embedding = self.text_encoder(input_ids, attn_mask, note_time_mask)

        if self.use_missing_embeds:
            dem_missing_mask = torch.all(dem == 0, dim=1)
            if dem_missing_mask.any(): dem_embedding[dem_missing_mask] = self.missing_dem_embedding.to(dem_embedding.device)
            
            ts_missing_mask = torch.all(ts_input.flatten(1) == 0, dim=1)
            if ts_missing_mask.any(): ts_embedding[ts_missing_mask] = self.missing_ts_embedding.to(ts_embedding.device)
            
            text_missing_mask = torch.all(input_ids.flatten(1) == 0, dim=1)
            if text_missing_mask.any(): text_embedding[text_missing_mask] = self.missing_text_embedding.to(text_embedding.device)

        if self.early_fusion_strategy == 'weighted':
            dem_proj = self.dem_fusion_proj(dem_embedding)
            ts_proj = self.ts_fusion_proj(ts_embedding)
            text_proj = self.text_fusion_proj(text_embedding)
            stacked_outputs = torch.stack([dem_proj, ts_proj, text_proj], dim=1)
            fusion_softmax_weights = F.softmax(self.fusion_weights, dim=0)
            fused_embedding = torch.sum(stacked_outputs * fusion_softmax_weights.view(1, -1, 1), dim=1)
        else: 
            fused_embedding = torch.cat([dem_embedding, ts_embedding, text_embedding], dim=1)
        
        projected_embedding = self.fusion_projection(fused_embedding)
        projected_embedding = self.fusion_layernorm(projected_embedding)
        
        
        gate_logits = -torch.cdist(projected_embedding, self.gating_centroids, p=2) 

        if self.training and self.is_dropout_enabled and self.dropped_expert_indices:
            gate_logits[:, self.dropped_expert_indices] = -1e9

        if self.training and self.noise_eps > 0:
            gate_logits += torch.randn_like(gate_logits) * self.noise_eps

        top_logits, top_indices = gate_logits.topk(self.k, dim=1)
        top_k_gates = F.softmax(top_logits / max(self.gating_temp, 1e-6), dim=1)
        gates = torch.zeros_like(gate_logits).scatter(1, top_indices, top_k_gates.to(gate_logits.dtype))
        
        if self.training and self.is_dropout_enabled:
            current_load = (gates > 0).sum(0).detach()
            self.expert_load_tracker += current_load

        moe_aux_loss = self._compute_moe_loss(gates)

        if self.training and step is not None and int(os.environ.get("RANK", "0")) == 0 and step % 200 == 0:
            load = (gates > 0).sum(0)
            print(f"--> Step {step} | EarlyFusionMoE | Expert Load (counts): {load.detach().cpu().numpy()}")

        dispatcher = SparseDispatcher(len(self.experts), gates)
        expert_inputs = dispatcher.dispatch(projected_embedding)
        expert_outputs = [self.experts[i](inp) for i, inp in enumerate(expert_inputs) if inp.numel() > 0]

        if expert_outputs:
            moe_output = dispatcher.combine(expert_outputs)
        else:
            moe_output = torch.zeros(
                projected_embedding.size(0), 
                self.config.moe_output_dim, 
                device=projected_embedding.device,
                dtype=projected_embedding.dtype
            )
        
       
        normalized_output = self.final_layernorm(moe_output)
        
        if self.classifier_head == 'advanced':
            x_proj = self.proj1(normalized_output)
            x_proj = F.relu(x_proj)
            x_proj = self.head_dropout(x_proj)
            x_proj = self.proj2(x_proj)
            final_projection = normalized_output + x_proj
            logits = self.out_layer(final_projection)
        else: 
            logits = self.classifier(normalized_output)

        total_aux_loss = (self.loss_coef * moe_aux_loss) + getattr(self.config, 'lambda_sparse', 1e-3) * M_loss
        
        return logits, total_aux_loss, expert_usage_counts

class TriMoE(nn.Module):
    def __init__(self, config, biobert_model, pretrained_centroids=None):
        super().__init__()
        self.config = config
        self.lambda_sparse = config.lambda_sparse

        
        self.dem_encoder = DemographicsTabNetEncoder(
            config.dem_input_dim,
            config.dem_embed_dim,
            config.tabnet_params
        )
        
        self.ts_encoder = TimeSeriesEncoder(
            input_dim=config.ts_input_dim,
            embed_dim=config.ts_embed_dim,
            embed_time=16,
            num_heads=2,
            attention_type=getattr(config, 'ts_attention_type', 'standard'),
            time_embedder_type=getattr(config, 'ts_time_embedder', 'simple'),
            tt_max=config.tt_max
        )

        self.text_encoder = TimeAwareTextEncoder(
            biobert_model,
            config.text_embed_dim,
            transformer_heads=8,
            transformer_layers=config.text_transformer_layers,
            use_residual_block=getattr(config, 'use_residual_text_block', False)
        )

        self.use_missing_embeds = getattr(config, 'use_missing_embeds', False)
        if self.use_missing_embeds:
            print("INFO: Using learnable embeddings for missing modalities.")
            self.missing_dem_embedding = nn.Parameter(torch.randn(config.dem_embed_dim))
            self.missing_ts_embedding = nn.Parameter(torch.randn(config.ts_embed_dim))
            self.missing_text_embedding = nn.Parameter(torch.randn(config.text_embed_dim))

        
        input_sizes_for_moe = [config.dem_embed_dim, config.ts_embed_dim, config.text_embed_dim]
        self.moe_layer = DisjointSparseMoE(
            num_modalities=3,
            input_sizes=input_sizes_for_moe,
            num_experts_per_modality=config.num_experts,
            expert_output_size=config.moe_output_dim,
            expert_hidden_size=config.moe_hidden_dim,
            k=config.top_k,
            loss_coef=config.moe_loss_coef,
            dropout_p=config.dropout_p,
            pretrained_centroids=pretrained_centroids,
            gating_temp=config.gating_temp,
            noise_eps=getattr(config, 'moe_noise_eps', 0.0),
            use_stabilized_expert=getattr(config, 'use_stabilized_expert', False),
            fusion_strategy=getattr(config, 'fusion_strategy', 'sum'),
            classifier_head=getattr(config, 'classifier_head', 'simple'),
            
            expert_dropout_min_k=getattr(config, 'expert_dropout_min_k', 0),
            expert_dropout_max_k=getattr(config, 'expert_dropout_max_k', 0),
            expert_dropout_persistence_prob=getattr(config, 'expert_dropout_persistence_prob', 0.0)
        )
    def clear_dropped_experts(self):
        """Convenience wrapper for the MoE layer."""
        if hasattr(self.moe_layer, 'clear_dropped_experts'):
            self.moe_layer.clear_dropped_experts()

    def update_dropped_experts(self):
        """
        A convenient wrapper to call the underlying MoE layer's update method.
        This makes the training loop cleaner.
        """
        if hasattr(self.moe_layer, 'update_dropped_experts'):
            self.moe_layer.update_dropped_experts()

    def forward(self, ts_input, ts_mask, ts_tt, dem, input_ids, attn_mask, note_time_mask, step=None):
        batch_size = dem.shape[0]

        
        dem_embedding, M_loss = self.dem_encoder(dem)
        ts_embedding = self.ts_encoder(ts_input, ts_tt, ts_mask)
        text_embedding = self.text_encoder(input_ids, attn_mask, note_time_mask)

        
        if self.use_missing_embeds:
            if torch.all(dem == 0, dim=1).any():
                missing_mask = torch.all(dem == 0, dim=1)
                dem_embedding[missing_mask] = self.missing_dem_embedding.to(dem_embedding.device)

            if torch.all(ts_input.flatten(1) == 0, dim=1).any():
                missing_mask = torch.all(ts_input.flatten(1) == 0, dim=1)
                ts_embedding[missing_mask] = self.missing_ts_embedding.to(ts_embedding.device)

            if torch.all(input_ids.flatten(1) == 0, dim=1).any():
                missing_mask = torch.all(input_ids.flatten(1) == 0, dim=1)
                text_embedding[missing_mask] = self.missing_text_embedding.to(text_embedding.device)

        modalities_list = [dem_embedding, ts_embedding, text_embedding]
        final_logits, moe_aux_loss, expert_usage = self.moe_layer(modalities_list, step=step)
        
        total_aux_loss = moe_aux_loss + self.lambda_sparse * M_loss
        return final_logits, total_aux_loss, expert_usage