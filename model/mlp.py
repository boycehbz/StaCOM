import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class AdaLN(nn.Module):
    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        emb_out = self.emb_layers(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h


class VanillaSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout, batch_first=True,
                                               add_zero_attn=True)

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        """
        x_norm = self.norm(x, emb)
        y = self.attention(x_norm, x_norm, x_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return y


class VanillaCrossAttention(nn.Module):
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, kdim=xf_latent_dim, vdim=xf_latent_dim,
                                               dropout=dropout, batch_first=True, add_zero_attn=True)

    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        x_norm = self.norm(x, emb)
        xf_norm = self.xf_norm(xf, emb)
        y = self.attention(x_norm, xf_norm, xf_norm,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return y


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class ObjectDenoiser(nn.Module):

    def __init__(self,
                 motion_dim,
                 obj_traj_dim=12,
                 obj_shape_dim=44,
                 latent_dim=512,
                 num_frames=280,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 **kwargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.motion_dim = motion_dim
        self.time_embed_dim = latent_dim

        # Object feature dimensions
        self.obj_traj_dim = obj_traj_dim
        self.obj_shape_dim = obj_shape_dim

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Motion Embedding
        self.motion_embed_a = nn.Linear(self.motion_dim//2, self.latent_dim) 
        self.motion_embed_b = nn.Linear(self.motion_dim//2, self.latent_dim)  
        
        # Object Embedding
        self.obj_traj_embed = nn.Linear(self.obj_traj_dim, self.latent_dim)
        self.obj_shape_embed = nn.Linear(self.obj_shape_dim, self.latent_dim)

        # Transformer blocks for interaction modeling
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(
                num_heads=num_heads,
                latent_dim=latent_dim, 
                dropout=dropout, 
                ff_size=ff_size))
        
        # Output Module
        self.out_a = zero_module(FinalLayer(self.latent_dim, self.motion_dim//2))  
        self.out_b = zero_module(FinalLayer(self.latent_dim, self.motion_dim//2)) 

    def forward(self, x, timesteps, obj_trajectory=None, obj_shape=None, mask=None, **kwargs):
        B, T, D = x.shape
        x_a = x[..., :self.motion_dim//2]  
        x_b = x[..., self.motion_dim//2:]  

        if mask is not None:
            mask = mask[..., 0]

        # Time embedding
        emb = self.embed_timestep(timesteps)  # [B, latent_dim]

        if obj_trajectory is not None:
            obj_traj_emb = self.obj_traj_embed(obj_trajectory)  # [B, T, latent_dim]
        else:
            obj_traj_emb = torch.zeros(B, T, self.latent_dim, device=x.device)
            
        if obj_shape is not None:
            obj_shape_emb = self.obj_shape_embed(obj_shape)  # [B, latent_dim]
            emb = emb + obj_shape_emb  # Add shape condition to time embedding
        
        # Motion embedding
        a_emb = self.motion_embed_a(x_a)  # [B, T, latent_dim]
        b_emb = self.motion_embed_b(x_b)  # [B, T, latent_dim]
        
        # Add object trajectory conditioning
        a_emb = a_emb + obj_traj_emb
        b_emb = b_emb + obj_traj_emb
        
        # Positional encoding
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x.device)
        key_padding_mask = ~(mask > 0.5)

        # Transformer blocks with cross-attention between two persons
        for i, block in enumerate(self.blocks):
            # Person A attends to Person B
            h_a = block(h_a_prev, h_b_prev, emb, key_padding_mask)
            # Person B attends to Person A
            h_b = block(h_b_prev, h_a_prev, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        # Output projection
        output_a = self.out_a(h_a)
        output_b = self.out_b(h_b)

        # Combine outputs
        output = torch.cat([output_a, output_b], dim=-1)

        return output


SimpleObjectConditionedDenoiser = ObjectDenoiser
InterGenObjectDenoiser = ObjectDenoiser
InterGenDenoiseNetwork = ObjectDenoiser