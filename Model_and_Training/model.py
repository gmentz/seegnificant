import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange

# Convolutional tokenizer
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=5):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 40), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.AvgPool2d((1, 40), (1, 5)),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        return x


# Multi-Headed Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            mask = mask[:, None, :, None] + mask[:, None, None, :] == 1  # Mask is 0 @ real channels and 1 @ padded channels.
            fill_value = torch.finfo(torch.float32).min  # Fill value = -inf
            energy = energy.masked_fill(mask, fill_value)  # Fill all the attn weights @ padded channels with -inf

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# Feed Forward Block used in transformer encoder
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, emb_size),
            nn.Dropout(drop_p),
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=1, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__()

        # Attention things
        self._attn_layer_norm = nn.LayerNorm(emb_size)
        self._attn = MultiHeadAttention(emb_size, num_heads, drop_p)
        self._attn_dropout = nn.Dropout(drop_p)

        # FFN things
        self._ffn_layer_norm = nn.LayerNorm(emb_size)
        self._ffn = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self._ffn_dropout = nn.Dropout(drop_p)

    def forward(self, x: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        res = x
        x = self._attn_layer_norm(x)
        x = self._attn(x, mask=mask)
        x = self._attn_dropout(x)
        x += res

        res = x
        x = self._ffn_layer_norm(x)
        x = self._ffn(x)
        x = self._ffn_dropout(x)
        x += res

        return x


# Subject-specific regression heads using for regression (can also be used for classification)
class TaskHeads(nn.Module):
    def __init__(self, emb_size, numParticipants=21):
        super().__init__()

        self.numPart = numParticipants  # To identify how many task heads to keep

        # self.clf = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.numPart)])  # Classification heads (not used for work in this manuscript)

        self.reg = nn.ModuleList([nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.ReLU()) for _ in range(self.numPart)])

    def forward(self, x, SubjId):

        out_reg = torch.zeros(len(SubjId), 1, device=x.device)
        for i in range(max(SubjId) + 1):
            xx = x[SubjId == i, :]
            out_reg[SubjId == i] = self.reg[i](xx)

        return out_reg


# Novel positional encoding based on radial basis functions.
class mniPositionalEncoding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

        means = torch.arange(start=-90, end=80, step=20, device='cuda')  # Specify the mean of the gaussians (grid)
        stds = torch.tensor([1, 2, 4, 8, 16, 32, 64], device='cuda')  # Specify the scale of the gaussian

        self.gaussians = torch.distributions.Normal(means.repeat_interleave(stds.size(0)), stds.repeat(means.size(0)))

        self.projection = nn.Linear(means.size(0) * stds.size(0) * 3, emb_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, mni_coords):

        # Calculate the gaussian features
        b, c, v = mni_coords.size()  # batch, electrode, MNI coordinate triplet (x, y, z)
        mni_coords = rearrange(mni_coords, "b c v -> (b c v)")
        mni_coords = torch.exp(self.gaussians.log_prob(mni_coords.unsqueeze(1)))
        mni_coords = rearrange(mni_coords, "(b c v) f -> b c (v f)", b=b, c=c, v=v)
        mni_coords = self.projection(mni_coords)
        mni_coords = self.dropout(mni_coords)
        mni_coords = rearrange(mni_coords, "b c f -> b f c 1")

        # Add the gaussian features to convolutional features.
        return x + mni_coords


class seegnificant(nn.Module):
    def __init__(self, emb_size, depth, numPart, **kwargs):
        super().__init__()

        # Tokenizer
        self.PatchEmbedding = PatchEmbedding(emb_size=emb_size)
        # Transformers
        self.depth = depth
        self.TimeTransformerEncoder = nn.ModuleList([TransformerEncoderBlock(emb_size) for _ in range(depth)])
        self.SpaceTransformerEncoder = nn.ModuleList([TransformerEncoderBlock(emb_size) for _ in range(depth)])
        self.mniPositionalEncoding = mniPositionalEncoding(emb_size)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(5880, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
            )

        self.taskHeads = TaskHeads(emb_size, numParticipants=numPart)

    def forward(self, x, subj_id, mni_coords, returnFeatures=False):
        '''
        :param x: batch x features x electrodes x timepoints
        :param subj_id: batch x 1
        :param mni_coords: batch x electrodes x 3
        :param returnFeatures: Bool
        :return:
        '''

        # Embedding Block
        x = self.PatchEmbedding(x)
        batch_size, num_filters, num_electr, num_timepoints = x.shape

        # Identify all the padded electrodes (all values across time are 0 for padded electrodes)
        padded_channels = mni_coords.sum(dim=-1) == 0  # 0 for all real channels & 1 for all padded channels
        padded_channels.unsqueeze_(-1)
        padded_channel_mask = padded_channels.repeat(1, 1, num_timepoints)
        padded_channel_mask = rearrange(padded_channel_mask, "(b) c (t) -> (b t) c")

        # Sequential Transformer Blocks
        for i in range(self.depth):
            x = rearrange(x, '(b) k (c) t -> (b c) t k')  # Squeeze the batch and channel dimensions
            x = self.TimeTransformerEncoder[i](x)
            x = rearrange(x, '(b c) t k -> b k c t', c=num_electr)
            x = self.mniPositionalEncoding(x, mni_coords)
            x = rearrange(x, '(b) k c t -> (b t) c k')
            x = self.SpaceTransformerEncoder[i](x, padded_channel_mask)
            x = rearrange(x, '(b t) c k-> b k c t', t=num_timepoints)

        x = x.contiguous().view(x.size(0), -1)

        # MLP to enhance fitting ability
        f = self.mlp(x)

        # Predict the RT
        reg = self.taskHeads(f, subj_id)

        if returnFeatures:
            return f, reg
        else:
            return reg
