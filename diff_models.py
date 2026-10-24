import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from utils.SelfAttention_Family import FullAttention, TV_AttentionLayer, TV_AttentionEncoderLayer, AttentionLayer, AttentionEncoderLayer

def get_torch_trans(heads=8, layers=1, channels=64, dropout=0.1):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", dropout=dropout
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def get_custom_tv_trans(heads=8, layers=1, channels=64, dropout=0.0, pre_norm=False):
    encoder_layer = TV_AttentionLayer(
        FullAttention(mask_flag=False),
        d_t=channels//4,
        d_v=channels,
        n_heads=heads
        )
    return TV_AttentionEncoderLayer(encoder_layer, d_t=channels//4, d_v=channels, d_ff=64, activation="gelu", dropout=dropout, pre_norm=pre_norm)

def get_cross_trans(heads=8, layers=1, channels=64, dropout=0.0, pre_norm=False):
    encoder_layer = AttentionLayer(
        FullAttention(mask_flag=False),
        d_model=channels,
        n_heads=heads
    )
    return AttentionEncoderLayer(encoder_layer, d_model=channels, d_ff=64, activation="gelu", dropout=dropout, pre_norm=pre_norm)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x) 
        x = F.silu(x)
        x = self.projection2(x) 
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1) 
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0) 
        table = steps * frequencies 
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  
        return table


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False, with_text=False, context_dim=None, dropout=0., attn_drop=0., pre_norm=False, pred_len=-1, use_tv=False, text_mode="legacy", tower_dropout=None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_drop)

        self.is_linear = is_linear
        self.text_mode = str(text_mode or "legacy").lower()
        if self.text_mode not in ("legacy", "gated", "off"):
            raise ValueError(f"unknown text_mode {text_mode}")
        self.with_text = bool(with_text) and self.text_mode != "off"
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            # Legacy blocks keep the historical 0.1 Transformer dropout unless a config
            # sets tower_dropout. xattn sets 0 for every text mode so A1 and T1 share a backbone.
            # Gated blocks without that key still use the configured dropout (CLI default 0).
            # 0 is a real value here; only a missing key falls through.
            if tower_dropout is None:
                tower_dropout = dropout if self.text_mode == "gated" else 0.1
            else:
                tower_dropout = float(tower_dropout)
            if use_tv:
                self.time_layer = get_custom_tv_trans(heads=nheads, layers=1, channels=channels, dropout=dropout, pre_norm=pre_norm)
            else:
                self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels, dropout=tower_dropout)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels, dropout=tower_dropout)
        self.channel_proj = None
        self.cross_modal_layer = None
        self.cross_attention = None
        self.query_norm = None
        self.key_norm = None
        self.text_alpha = None
        if self.with_text and self.text_mode == "legacy":
            self.channel_proj = nn.Linear(context_dim, channels)
            self.cross_modal_layer = get_cross_trans(heads=nheads, layers=1, channels=channels, dropout=dropout, pre_norm=pre_norm)
        elif self.with_text and self.text_mode == "gated":
            # No per-block channel projection and no FFN. Attention dropout is 0 so a
            # zero gate is an exact no-op and the benefit label is not dropout noise.
            self.query_norm = nn.LayerNorm(channels)
            self.key_norm = nn.LayerNorm(channels)
            self.cross_attention = AttentionLayer(
                FullAttention(mask_flag=False, attention_dropout=0.0, output_attention=False),
                d_model=channels,
                n_heads=nheads,
            )
            self.text_alpha = nn.Parameter(torch.zeros(1))

    # fusing information along L
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    # fusing information along K
    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward_time_TV(self, yt, yv, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return yt, yv
        yt = yt.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)
        yv = yv.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)

        yt, yv, attn = self.time_layer(yt, yv)

        yt = yt.reshape(B, K, -1, L).permute(0, 2, 1, 3).reshape(B, -1, K * L)
        yv = yv.reshape(B, K, -1, L).permute(0, 2, 1, 3).reshape(B, -1, K * L)
        return yt, yv, attn

    def forward_time_TV_nosep(self, yt, yv, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return yt, yv
        yt = yt.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)
        yv = yv.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)

        y = torch.cat([yt, yv], dim=1)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def _apply_gated_cross_attention(self, y, text_memory, text_token_mask, text_inject, return_text_stats):
        batch, channels, positions = y.shape
        queries = self.query_norm(y.transpose(1, 2))
        keys = self.key_norm(text_memory)
        text_stats = None
        if return_text_stats:
            query_bar = self.cross_attention.query_projection(queries).mean(dim=1)
            key_projected = self.cross_attention.key_projection(keys)
            if text_token_mask is None:
                key_bar = key_projected.mean(dim=1)
                any_valid = torch.ones((batch,), device=y.device, dtype=y.dtype)
            else:
                token_mask = text_token_mask.to(device=y.device, dtype=y.dtype)
                denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                key_bar = (key_projected * token_mask.unsqueeze(-1)).sum(dim=1) / denom
                any_valid = (token_mask.sum(dim=1) > 0).to(dtype=y.dtype)
            text_stats = {"q_bar": query_bar, "k_bar": key_bar, "any_valid": any_valid}

        if text_inject is None:
            inject = torch.ones((batch,), device=y.device, dtype=y.dtype)
        else:
            inject = text_inject.to(device=y.device, dtype=y.dtype).reshape(batch)
        if text_token_mask is None:
            token_valid = torch.ones((batch, text_memory.shape[1]), device=y.device, dtype=torch.bool)
            row_valid = torch.ones((batch,), device=y.device, dtype=y.dtype)
        else:
            token_valid = text_token_mask.to(device=y.device).bool()
            row_valid = token_valid.any(dim=-1).to(dtype=y.dtype)
        attn_out, _ = self.cross_attention(
            queries,
            keys,
            keys,
            attn_mask=None,
            key_padding_mask=token_valid,
        )
        scale = inject * row_valid * torch.tanh(self.text_alpha).to(dtype=y.dtype)
        delta = attn_out.transpose(1, 2) * scale.view(batch, 1, 1)
        return y + delta, text_stats

    def forward(self, x, cond_info, diffusion_emb, timesteps_emb=None, context=None, text_memory=None, text_token_mask=None, text_inject=None, return_text_stats=False):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1) 
        y = x + diffusion_emb 

        atten_series_timestamp = None
        if self.is_linear:
            y = self.forward_time(y, base_shape) 
        elif timesteps_emb is not None:
            timesteps_emb, y, atten_series_timestamp = self.forward_time_TV(timesteps_emb, y, base_shape)
        else:
            y = self.forward_time(y, base_shape) 
        y = self.forward_feature(y, base_shape) 
        
        attn_series_text = None
        text_stats = None
        if self.text_mode == "gated" and text_memory is not None:
            y, text_stats = self._apply_gated_cross_attention(
                y, text_memory, text_token_mask, text_inject, return_text_stats
            )
        elif self.with_text and self.text_mode == "legacy" and context is not None:
            context = self.channel_proj(context.permute(0,2,1)).permute(0,2,1) 
            y, attn_series_text = self.cross_modal_layer(y, context) 

        y = self.attn_drop(y)

        y = self.mid_projection(y)
        y = self.dropout(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info 

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  
        y = self.output_projection(y) 
        y = self.dropout(y)

        residual, skip = torch.chunk(y, 2, dim=1) 
        x = x.reshape(base_shape) 
        residual = residual.reshape(base_shape) 
        skip = skip.reshape(base_shape) 
        output = ((x + residual) / math.sqrt(2.0), skip, timesteps_emb, atten_series_timestamp, attn_series_text)
        if return_text_stats:
            return output + (text_stats,)
        return output

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2, mode_num=0):
        super().__init__()
        self.channels = config["channels"]
        self.mode_num = mode_num
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.lookback_len = config["lookback_len"]
        self.pred_len = config["pred_len"]
        self.with_timestep = config["with_timestep"]
        dropout = config["dropout"]
        attn_drop = config["attn_drop"]
        self.pre_norm = config["pre_norm"]
        self.time_weight= config["time_weight"]
        self.save_attn = config["save_attn"]
        self.text_mode = str(config.get("text_mode", "legacy")).lower()
        self.n_text_sources = int(config.get("n_text_sources", 2))

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.dropout = nn.Dropout(dropout)

        if self.with_timestep:
            self.timestep_projection1 = Conv1d_with_init(self.channels//4, self.channels//4, 1)
            self.timestep_projection2 = Conv1d_with_init(self.channels//4, 1, 1)
            nn.init.zeros_(self.timestep_projection2.weight)
            self.weight_mlp = nn.Sequential(
                nn.Linear(self.lookback_len, 16),
                nn.GELU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
            )   

        self.text_proj = None
        self.text_source_emb = None
        if config.get("with_texts", False) and self.text_mode == "gated":
            self.text_proj = nn.Linear(config["context_dim"], self.channels)
            self.text_source_emb = nn.Embedding(self.n_text_sources, self.channels)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    with_text=config["with_texts"],
                    context_dim=config["context_dim"],
                    dropout=dropout,
                    attn_drop=attn_drop,
                    pre_norm=self.pre_norm,
                    pred_len=self.pred_len,
                    use_tv=config.get("with_timestep", False),
                    text_mode=self.text_mode,
                    tower_dropout=config.get("tower_dropout", None),
                )
                for _ in range(config["layers"])
            ]
        )

    def _project_text_memory(self, text_hidden, text_source_ids):
        if text_hidden is None or self.text_proj is None:
            return None
        memory = self.text_proj(text_hidden)
        if text_source_ids is not None and self.text_source_emb is not None:
            source_emb = self.text_source_emb(text_source_ids.to(device=memory.device))
            if source_emb.dim() == 2:
                source_emb = source_emb.unsqueeze(0)
            memory = memory + source_emb
        return memory

    def forward(self, x, cond_info, diffusion_step, cfg_mask, timestep_emb=None, size_emb=None, context=None, text_hidden=None, text_token_mask=None, text_source_ids=None, return_text_stats=False):
        B, inputdim, K, L = x.shape
        obs = x[:, 0, :, :self.lookback_len]
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x) 
        x = self.dropout(x)
        x = F.relu(x)
        if timestep_emb is not None:
            timestep_emb = timestep_emb.reshape(B, -1, K*L)
        if size_emb is not None:
            size_emb = size_emb.reshape(B, -1, K*L)
            x = torch.cat([x,  size_emb], dim=1)
        text_memory = None
        text_inject = None
        if self.text_mode == "gated":
            text_memory = self._project_text_memory(text_hidden, text_source_ids)
            text_inject = cfg_mask
        elif context is not None and cfg_mask is not None:
            cfg_mask = cfg_mask[:, None, None].repeat(1, context.shape[1], context.shape[2])
            context = context * cfg_mask
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step) 

        skip = []
        timestep_emb_ls = []
        attn = []
        first_text_stats = None
        for layer_index, layer in enumerate(self.residual_layers):
            want_stats = bool(return_text_stats and layer_index == 0 and text_memory is not None)
            layer_out = layer(
                x,
                cond_info,
                diffusion_emb,
                timestep_emb,
                context,
                text_memory=text_memory,
                text_token_mask=text_token_mask,
                text_inject=text_inject,
                return_text_stats=want_stats,
            )
            if want_stats:
                x, skip_connection, timestep_emb, attn_time, attn_text, first_text_stats = layer_out
            else:
                x, skip_connection, timestep_emb, attn_time, attn_text = layer_out
            skip.append(skip_connection)
            timestep_emb_ls.append(timestep_emb)
            attn.append((attn_time, attn_text))

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L) 
        x = self.output_projection1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)

        if timestep_emb is not None:
            timestep_pred = torch.sum(torch.stack(timestep_emb_ls), dim=0) / math.sqrt(len(self.residual_layers)) 
            timestep_pred = timestep_pred.reshape(B, self.channels//4, K * L)
            timestep_pred = self.timestep_projection1(timestep_pred)
            timestep_pred = F.relu(timestep_pred)
            timestep_pred = self.timestep_projection2(timestep_pred)
            timestep_pred = timestep_pred.reshape(B, K, L)

            error = timestep_pred[:, :, :self.lookback_len] - obs
            conb_w = self.weight_mlp(error).unsqueeze(2) 
            x = torch.stack([x, timestep_pred], dim=-1) 
            x = torch.sum(x * conb_w, dim=-1) 
        similarity = None
        if return_text_stats:
            if first_text_stats is None:
                similarity = x.new_zeros((B,))
            else:
                similarity = F.cosine_similarity(
                    first_text_stats["q_bar"].float(),
                    first_text_stats["k_bar"].float(),
                    dim=-1,
                    eps=1e-8,
                )
                similarity = similarity * first_text_stats["any_valid"].to(dtype=similarity.dtype)
        if self.save_attn and return_text_stats:
            return x, attn, similarity
        if return_text_stats:
            return x, similarity
        if self.save_attn:
            return x, attn 
        else:
            return x
