# model
import os
import random
from typing import List
import torch
from torch import nn
import torch.nn.functional as F

class Expert(nn.Module):
    """basic mlp expert with ReLU activation"""
    def __init__(self, input_size, output_size, hidden_size, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)



class StabilizedExpert(nn.Module):
    """more stable MLP expert with LayerNorm, GELU, and xavier init"""
    def __init__(self, input_size, output_size, hidden_size, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Dropout(dropout_p), nn.Linear(hidden_size, output_size))
        
        self._init_weights()
    
    # xavier init works well with GELU 
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)


  
class SparseDispatcher:
    def __init__(self, num_experts: int, gates: torch.Tensor):
        """to enforce routing logic - adapted from Han et al (2024)
        well explained as not easy to understand"""
        self._gates = gates
        self._num_experts = num_experts

        # find positions (that are not zero)
        self._batch_idx, self._expert_idx = torch.where(gates > 0)
        # sort by expert index so when splitting it matches the expert block
        sorted_expert_idx, self._sorted_indices = self._expert_idx.sort(0)
        self._batch_idx = self._batch_idx[self._sorted_indices]
        # keep the gate values in the order based on the expert index
        self._nonzero_gates = gates[self._batch_idx, sorted_expert_idx]
        # how many examples go to each expert
        self._part_sizes = torch.bincount(sorted_expert_idx, minlength=num_experts).tolist()
        #  list created of where to route what

    def dispatch(self, inp: torch.Tensor) -> List[torch.Tensor]:
        # checks on correct device to avoid crash
        if inp.device != self._nonzero_gates.device:
            inp = inp.to(self._nonzero_gates.device)
        # error check in case of empty batch scenario - returns 0 tensors of right shape
        if self._batch_idx.numel() == 0:
            return [torch.zeros(0, inp.size(1), device=inp.device, dtype=inp.dtype) for _ in range(self._num_experts)]
        # pulls the data in the correct order based on expert
        gathered = inp[self._batch_idx]
        # splits the data based on expert to be sent to
        return list(torch.split(gathered, self._part_sizes, dim=0))

    def combine(self, expert_out: List[torch.Tensor], multiply_by_gates: bool = True) -> torch.Tensor:
        # if no input, produce 0 filled output
        if len(expert_out) == 0:
            return torch.zeros(self._gates.size(0), 0, device=self._gates.device)
        # combines the data into one large tensor of the expert ouputs (still grouped by expert)
        stitched_out = torch.cat(expert_out, 0)  
        # multiply by gating confidence
        if multiply_by_gates:
            stitched_out = stitched_out.mul(self._nonzero_gates.unsqueeze(1))
        # sorts back into orignal position from init method
        zeros_like_stitched = torch.zeros_like(stitched_out)
        zeros_like_stitched[self._sorted_indices] = stitched_out
        # creates the output tensor wiith all zeros
        out_dim = expert_out[0].size(1)
        zeros = torch.zeros(self._gates.size(0), out_dim, device=stitched_out.device, dtype=stitched_out.dtype)
        # uses stored positions from original_batch_idx for each routed example and adds together weighted expert outputs
        original_batch_idx, _ = torch.where(self._gates > 0)
        combined = zeros.index_add(0, original_batch_idx, zeros_like_stitched)
        return combined    


# DISJOINT

class DisjointSparseMoE(nn.Module):
    def __init__(self,
        num_modalities: int,
        input_sizes: List[int],
        num_experts_per_modality: List[int],
        expert_output_size: int,
        expert_hidden_size: int,
        k: int,
        loss_coef: float,
        dropout_p: float = 0.2,
        pretrained_centroids=None,
        gating_temp: float = 1.0,
        noise_eps: float = 0.1,
        use_stabilized_expert: bool = False,
        fusion_strategy: str = 'sum',
        classifier_head: str = 'simple',
        
        expert_dropout_min_k: int = 0,
        expert_dropout_max_k: int = 0,
        expert_dropout_persistence_prob: float = 0.0):

        super().__init__()
        assert num_modalities == len(input_sizes)
        assert num_modalities == len(num_experts_per_modality), "provide an expert count for each modality."
        
        self.num_modalities = num_modalities
        self.num_experts_per_modality = num_experts_per_modality
        self.k = k
        self.loss_coef = loss_coef
        self.gating_temp = gating_temp
        self.expert_output_size = expert_output_size
        self.noise_eps = noise_eps
        self.fusion_strategy = fusion_strategy
        self.classifier_head = classifier_head

        
        self.expert_dropout_min_k = expert_dropout_min_k
        self.expert_dropout_max_k = expert_dropout_max_k
        self.expert_dropout_persistence_prob = expert_dropout_persistence_prob
        self.is_dropout_enabled = self.expert_dropout_min_k > 0 and self.expert_dropout_max_k >= self.expert_dropout_min_k
        
        if self.is_dropout_enabled:
            print(f"initializing CED and dropping {self.expert_dropout_min_k}-{self.expert_dropout_max_k} experts per modality each epoch.")
            print(f"persistence probability for top expert: {self.expert_dropout_persistence_prob:.2f}")
            
            self.expert_load_trackers = [torch.zeros(n_exp) for n_exp in self.num_experts_per_modality]
            
            self.dropped_expert_indices = [[] for _ in range(num_modalities)]
            
            self.experts_to_persist_drop = [None] * num_modalities
        

        if self.fusion_strategy == 'weighted':
            print("using learnable weighted sum for modality fusion.")
            self.fusion_weights = nn.Parameter(torch.ones(num_modalities))
        else:
            print("using simple summation for modality fusion.")

        if pretrained_centroids is not None:
            self.gating_centroids = nn.ParameterList([nn.Parameter(c.clone().detach().float()) for c in pretrained_centroids])
            print("initialised gating centroids from pre-training")
        else:
            print("using random initialisation for gating centroids")
            self.gating_centroids = nn.ParameterList([nn.Parameter(torch.randn(num_experts, size) * 0.4)
                for num_experts, size in zip(self.num_experts_per_modality, input_sizes)])
        
        ExpertChoice = StabilizedExpert if use_stabilized_expert else Expert
        self.experts = nn.ModuleList([
            nn.ModuleList([ExpertChoice(size, expert_output_size, hidden_size=expert_hidden_size, dropout_p=dropout_p
                ) for _ in range(num_experts)]) for num_experts, size in zip(self.num_experts_per_modality, input_sizes)])
        
        self.fused_layernorm = nn.LayerNorm(expert_output_size)
        
        if self.classifier_head == 'advanced':
            print("using a non-linear classification head")
            self.proj1 = nn.Linear(expert_output_size, expert_output_size)
            self.proj2 = nn.Linear(expert_output_size, expert_output_size)
            self.head_dropout = nn.Dropout(dropout_p)
            self.out_layer = nn.Linear(expert_output_size, 1)
        else:
            print("using a simple linear classification head")
            self.classifier = nn.Linear(expert_output_size, 1)

    
    def update_dropped_experts(self):
        """
        implements CED with stochasticity and persistence, as described in paper.
        called by the training loop at the end of each epoch.
        """
        if not self.is_dropout_enabled:
            return

        for i in range(self.num_modalities):
            
            if self.expert_load_trackers[i].sum() == 0:
                self.dropped_expert_indices[i] = []
                continue

            
            most_used_expert_this_epoch = torch.argmax(self.expert_load_trackers[i]).item()
            
            
            num_to_drop = random.randint(self.expert_dropout_min_k, self.expert_dropout_max_k)
            
            
            new_dropped_set = set()
            new_dropped_set.add(most_used_expert_this_epoch)
            
            
            if self.experts_to_persist_drop[i] is not None:
                new_dropped_set.add(self.experts_to_persist_drop[i])
            num_experts_total = self.num_experts_per_modality[i]
            
            k_for_topk = min(num_to_drop, num_experts_total)
            _, top_indices = torch.topk(self.expert_load_trackers[i], k_for_topk)
            
            for idx in top_indices.tolist():
                if len(new_dropped_set) >= num_to_drop:
                    break
                new_dropped_set.add(idx)

            self.dropped_expert_indices[i] = list(new_dropped_set)
                        
            if random.random() < self.expert_dropout_persistence_prob:
                self.experts_to_persist_drop[i] = most_used_expert_this_epoch
            else:
                self.experts_to_persist_drop[i] = None

            # logging
            modality_name = ["Demographics", "Time-Series", "Text"][i] if i < 3 else f"Modality-{i}"
            print(f" modality '{modality_name}': dropping experts {self.dropped_expert_indices[i]} for the next epoch")
            if self.experts_to_persist_drop[i] is not None:
                print(f"  expert ({self.experts_to_persist_drop[i]}) will be persisted for an extra epoch")

            
            self.expert_load_trackers[i].zero_()

    def clear_dropped_experts(self):
        if self.is_dropout_enabled:
            print("clearing all dropped experts all experts are now active for eval")
            self.dropped_expert_indices = [[] for _ in range(self.num_modalities)] 
            self.experts_to_persist_drop = [None] * self.num_modalities


    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.sum() <= eps: return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _compute_aux_loss(self, gates: torch.Tensor) -> torch.Tensor:
        load = (gates > 0).sum(0)
        importance = gates.sum(0)
        return self.cv_squared(load) + self.cv_squared(importance)

    def forward(self, x: list, step=None):
        device = x[0].device
        batch_size = x[0].size(0)
        dtype = x[0].dtype
        
        modality_outputs = []
        total_aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        expert_usage_counts = []

        for i in range(self.num_modalities):
            gate_logits = -torch.cdist(x[i], self.gating_centroids[i], p=2)

            
            if self.training and self.is_dropout_enabled and self.dropped_expert_indices[i]:
                indices_to_drop = self.dropped_expert_indices[i]
                gate_logits[:, indices_to_drop] = -1e9
            
            if self.training and self.noise_eps > 0:
                gate_logits += torch.randn_like(gate_logits) * self.noise_eps
            
            top_logits, top_indices = gate_logits.topk(self.k, dim=1)
            top_k_gates = F.softmax(top_logits / max(self.gating_temp, 1e-6), dim=1)
            gates = torch.zeros_like(gate_logits).scatter(1, top_indices, top_k_gates.to(gate_logits.dtype))
            
            
            if self.training and self.is_dropout_enabled:
                if self.expert_load_trackers[i].device != gates.device:
                    self.expert_load_trackers[i] = self.expert_load_trackers[i].to(gates.device)
                
                # sum of gating weights not count of selections
                
                current_load = gates.sum(0).detach()
                self.expert_load_trackers[i] += current_load
            
            total_aux_loss += self._compute_aux_loss(gates)
            
            load_counts = (gates > 0).sum(0)
            expert_usage_counts.append(load_counts.detach()) # detach to prevent gradient flow
            
            if step is not None and int(os.environ.get("RANK", "0")) == 0 and step % 200 == 0:
                load_counts = (gates > 0).sum(0) 
                modality_name = ["Demographics", "Time-Series", "Text"][i] if i < 3 else f"Modality-{i}"
                print(f"step = {step} , modality = {modality_name} , expert load = {load_counts.detach().cpu().numpy()}")

            dispatcher = SparseDispatcher(self.num_experts_per_modality[i], gates)
            expert_inputs = dispatcher.dispatch(x[i])
            expert_outputs = [self.experts[i][j](inp) for j, inp in enumerate(expert_inputs) if inp.numel() > 0]

            if expert_outputs:
                combined = dispatcher.combine(expert_outputs)
                modality_outputs.append(combined)
            else:
                modality_outputs.append(torch.zeros(batch_size, self.expert_output_size, device=device, dtype=dtype))

        if self.fusion_strategy == 'weighted':
            stacked_outputs = torch.stack(modality_outputs, dim=1)
            fusion_softmax_weights = F.softmax(self.fusion_weights, dim=0)
            fused_expert_outputs = torch.sum(
                stacked_outputs * fusion_softmax_weights.view(1, -1, 1), 
                dim=1
            )
        else:
            fused_expert_outputs = torch.sum(torch.stack(modality_outputs, dim=0), dim=0)

        normalized_fused_outputs = self.fused_layernorm(fused_expert_outputs)
        
        if self.classifier_head == 'advanced':
            x_proj = self.proj1(normalized_fused_outputs)
            x_proj = F.relu(x_proj)
            x_proj = self.head_dropout(x_proj)
            x_proj = self.proj2(x_proj)
            final_projection = normalized_fused_outputs + x_proj
            final_logits = self.out_layer(final_projection)
        else: 
            final_logits = self.classifier(normalized_fused_outputs)
        
        return final_logits, self.loss_coef * total_aux_loss, expert_usage_counts    

