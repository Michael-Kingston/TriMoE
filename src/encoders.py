# encoders.py
import math
import torch
from torch import nn
import torch.nn.functional as F

try:
    from pytorch_tabnet.tab_network import TabNetEncoder
except ImportError:
    print("pytorch-tabnet not installed")
    exit()




class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_layer = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, time_stamps):
        return self.embed_layer(time_stamps.unsqueeze(-1))
    

class Time2Vec(nn.Module):
    """Time2Vec from Kazemi et al (2019)"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.periodic = nn.Linear(1, embed_dim - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, time_stamps: torch.Tensor):
        tt = time_stamps.unsqueeze(-1) 
        out_periodic = torch.sin(self.periodic(tt))
               
        out_linear = self.linear(tt)
        
        return torch.cat([out_linear, out_periodic], -1)



class TimeAwareTextEncoder(nn.Module):
    def __init__(self, biobert_model, final_embed_dim, transformer_heads=8,
                 transformer_layers=2, dropout=0.1,
                 use_residual_block: bool = False): 
        super().__init__()
        self.bert = biobert_model
        bert_hidden_dim = self.bert.config.hidden_size

        self.note_sequence_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=bert_hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=bert_hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=transformer_layers)

        self.final_projection = nn.Linear(bert_hidden_dim, final_embed_dim)
        self.no_note_embedding = nn.Parameter(torch.zeros(bert_hidden_dim))
        self.layernorm = nn.LayerNorm(final_embed_dim)

        
        self.use_residual_block = use_residual_block
        if self.use_residual_block:
            print("text encoder will use a residual block for refinement")
            self.proj1 = nn.Linear(final_embed_dim, final_embed_dim)
            self.proj2 = nn.Linear(final_embed_dim, final_embed_dim)
            self.residual_dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, note_padding_mask):
        batch_size, num_notes, seq_len = input_ids.shape
        device = input_ids.device

        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attn = attention_mask.view(-1, seq_len)

        bert_output = self.bert(input_ids=flat_input_ids, attention_mask=flat_attn)
        notes_embedding = bert_output.last_hidden_state[:, 0, :].view(batch_size, num_notes, -1)

        has_notes_mask = (note_padding_mask > 0)
        src_key_padding_mask = ~has_notes_mask

        no_notes_in_sample = has_notes_mask.sum(dim=1) == 0
        if no_notes_in_sample.any():
            src_key_padding_mask[no_notes_in_sample, 0] = False

        contextualized_notes = self.note_sequence_transformer(src=notes_embedding, src_key_padding_mask=src_key_padding_mask)

        seq_lens = note_padding_mask.sum(dim=1)
        last_idx = torch.clamp(seq_lens - 1, min=0).long()
        last_note_emb = contextualized_notes[torch.arange(batch_size, device=device), last_idx]

        if no_notes_in_sample.any():
            last_note_emb[no_notes_in_sample] = self.no_note_embedding.to(device)

        projected = self.final_projection(last_note_emb)
        base_embedding = self.layernorm(projected)

        
        if self.use_residual_block:
            residual_input = base_embedding
            x = self.proj1(residual_input)
            x = F.relu(x)
            x = self.residual_dropout(x)
            x = self.proj2(x)
            
            final_embedding = residual_input + x
        else:
            final_embedding = base_embedding

        return final_embedding


class multiTimeAttention(nn.Module):
    """
    mTAND from Shukla et al ("2021) 
    """
    def __init__(self, value_feature_dim, nhidden, embed_time, num_heads):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.nhidden = nhidden

        
        self.linears = nn.ModuleList([
            nn.Linear(embed_time, embed_time),
            nn.Linear(embed_time, embed_time),
            nn.Linear(value_feature_dim * num_heads, nhidden)
        ])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        
        d_k = query.size(-1)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        num_features = value.size(-1)
        
        scores = scores.unsqueeze(-1).repeat_interleave(num_features, dim=-1)

        
        if mask is not None:
            
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1) == 0, -10000.0)
        
        p_attn = F.softmax(scores, dim=-2)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
               
        return torch.sum(p_attn * value.unsqueeze(2), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        batch_size = value.size(0)

        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears[:2], (query, key))]
        
       
        x, _ = self.attention(query, key, value, mask, dropout=F.dropout if dropout > 0 else None)
        
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * value.size(-1))
        
        return self.linears[-1](x)
    


class TimeSeriesEncoder(nn.Module):
    """
    GRU to tend to mTAND output 
    """
    def __init__(self, input_dim: int, embed_dim: int, embed_time: int, num_heads: int,
                 attention_type: str = 'standard',
                 time_embedder_type: str = 'simple',
                 tt_max: int = 48):
        super().__init__()
        self.attention_type = attention_type
        self.embed_dim = embed_dim
        
        if time_embedder_type == 'time2vec':
            print("timeSsriesencoder is using Time2Vec")
            self.time_embedder = Time2Vec(embed_dim=embed_time)
        else:
            print("timeseriesencoder is using a simple lnear layer for time embedding")
            self.time_embedder = TimeEmbedding(embed_dim=embed_time)

        if self.attention_type == 'irregular_aware':
            print("using MTAND with GRU summarizer")
            time_query_tensor = torch.linspace(0, 1., tt_max)
            self.register_buffer('time_query', time_query_tensor)
            
            self.attention_layer = multiTimeAttention(
                value_feature_dim=input_dim * 2, 
                nhidden=embed_dim, 
                embed_time=embed_time, 
                num_heads=num_heads)
            
            self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=False)
            
            
            self.final_proj = nn.Linear(embed_dim, embed_dim)

        else: 
            print("using standard attention encoder.")
            self.attention_layer = multiTimeAttention(
                value_feature_dim=input_dim, 
                nhidden=embed_dim,
                embed_time=embed_time,
                num_heads=num_heads
            )
            self.gru = None

    def forward(self, ts_values, ts_times, ts_mask):
        if ts_mask.dim() == 2:
            attention_mask_3d = ts_mask.unsqueeze(-1).expand_as(ts_values)
        else:
            attention_mask_3d = ts_mask

        time_embeddings = self.time_embedder(ts_times)
        per_timestep_mask = attention_mask_3d[:, :, 0]

        if self.attention_type == 'irregular_aware':
            query = self.time_embedder(self.time_query.unsqueeze(0).expand(ts_times.size(0), -1))
            key = time_embeddings
            
            value_with_mask = torch.cat([ts_values, attention_mask_3d.float()], dim=2)
            
            contextualized_sequence = self.attention_layer(query, key, value_with_mask, per_timestep_mask)
            
            gru_output, gru_hidden = self.gru(contextualized_sequence)
            
            final_hidden_state = gru_hidden.squeeze(0) 
            
            final_embedding = self.final_proj(final_hidden_state)

        else: 
            query = time_embeddings
            key = time_embeddings
            value = ts_values
            
            contextualized_sequence = self.attention_layer(query, key, value, per_timestep_mask)
    
            sequence_lengths = torch.clamp(per_timestep_mask.sum(dim=1) - 1, min=0).long()
            final_embedding = contextualized_sequence[
                torch.arange(ts_values.size(0), device=ts_values.device), 
                sequence_lengths
            ]
    
        return final_embedding
    
class DemographicsTabNetEncoder(nn.Module):
    def __init__(self, input_dim, final_output_dim, tabnet_params):
        super().__init__()
        self.n_d = tabnet_params.get("n_d", 8)
        self.tabnet = TabNetEncoder(input_dim=input_dim, output_dim=self.n_d + tabnet_params.get("n_a", 8), **tabnet_params)
        self.projection = nn.Linear(self.n_d, final_output_dim)

    def forward(self, x):
        device = x.device
        attn = self.tabnet.group_attention_matrix
        if attn.device != device:
            # move internal tensor to the correct device
            self.tabnet.group_attention_matrix = attn.to(device)
        step_outputs, M_loss = self.tabnet(x)
        combined = torch.sum(torch.stack(step_outputs, dim=0), dim=0)
        return self.projection(combined), M_loss
 
