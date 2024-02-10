import numpy as np
import torch
import torch.nn as nn

from modules.ATA import ATA
from modules.Autoformer import AutoCorrelation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_hid, max_len=1000):

        super(PositionalEncoding, self).__init__()
        # Create a long enough `P`
        d_hid = d_hid * 2
        self.P = torch.zeros((1, max_len, d_hid)).to(device)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_hid, 2, dtype=torch.float32) / d_hid)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        print(self.P.shape)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, attn_type):

        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.device = device

        self.d_model = d_model
        self.d_k = int(d_model / n_heads)
        self.n_heads = n_heads
        self.attn_type = attn_type

    def forward(self, Q, K, V):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ATA forecasting model

        if self.attn_type == "ATA":
            context, attn = ATA(d_k=self.d_k, device=self.device, h=self.n_heads)(
            Q=q_s, K=k_s, V=v_s)

        elif self.attn_type == "autoformer":
            context, attn = AutoCorrelation(seed=self.seed)(q_s.transpose(1, 2),
                                                            k_s.transpose(1, 2),
                                          v_s.transpose(1, 2))

        else:
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
            attn = torch.softmax(scores, -1)
            context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
            return context, attn

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        outputs = self.fc(context)
        return outputs


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, attn_type):

        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads,
            attn_type=attn_type)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads,
            attn_type=attn_type)
        self.pos_ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                     nn.ReLU(),
                                     nn.Linear(d_model*4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, dec_inputs, enc_outputs):

        out = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        out = self.norm1(dec_inputs + out)
        out2 = self.dec_enc_attn(out, enc_outputs, enc_outputs)
        out2 = self.norm2(out + out2)
        out3 = self.pos_ffn(out2)
        out3 = self.norm3(out2 + out3)
        return out3


class Decoder(nn.Module):

    def __init__(self, d_model,
                 n_heads, n_layers, attn_type):
        super(Decoder, self).__init__()

        self.device = device
        self.attn_type = attn_type
        self.pos_emb = PositionalEncoding(d_hid=d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                          n_heads=n_heads,
                                                          attn_type=attn_type) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = self.pos_emb(dec_inputs)

        for layer in self.layers:
            dec_outputs = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs)

        return dec_outputs


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, attn_type):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads,
            attn_type=attn_type)
        self.pos_ffn = nn.Sequential(nn.Linear(d_model, d_model*4),
                                     nn.ReLU(),
                                     nn.Linear(d_model*4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, enc_inputs):

        out = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        out = self.norm1(out + enc_inputs)
        out_2 = self.pos_ffn(out)
        out_2 = self.norm2(out_2 + out)
        return out_2


class Encoder(nn.Module):

    def __init__(self, d_model, n_heads,
                 n_layers, attn_type):
        super(Encoder, self).__init__()

        self.attn_type = attn_type
        self.pos_emb = PositionalEncoding(
            d_hid=d_model)
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(
                d_model=d_model, n_heads=n_heads, attn_type=attn_type) for _ in range(n_layers)])

    def forward(self, enc_input):

        enc_outputs = self.pos_emb(enc_input)

        for layer in self.layers:
            enc_outputs = layer(enc_outputs)

        return enc_outputs


class Transformer(nn.Module):

    def __init__(self, *, d_model=512, n_heads=8, n_layers=6, attn_type):
        super(Transformer, self).__init__()

        self.attn_type = attn_type

        self.encoder = Encoder(
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, attn_type=attn_type)
        self.decoder = Decoder(
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers,
            attn_type=attn_type)

        self.attn_type = attn_type

    def forward(self, enc_inputs):

        enc_outputs = self.encoder(enc_inputs)

        return enc_outputs
