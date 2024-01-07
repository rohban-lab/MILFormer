# most of the credit to http://nlp.seas.harvard.edu/annotated-transformer/ (We did serious changes in the attention and add more classes).
import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
import numpy as np
import math


def softmax_normal(x, dim, dtype=torch.float32):
    e_x = torch.exp(torch.sub(x, torch.max(x, dim=-1, keepdim=True)[0])).to(dtype)
    sum_e_x = torch.sum(e_x, dim=dim, keepdims=True, dtype=dtype)
    ret = torch.div(e_x, sum_e_x)
    return ret


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BClassifier(nn.Module):
    def __init__(self, encoder, num_classes, input_size: int):
        super(BClassifier, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        x, attentions = self.encoder(x, c)
        return self.linear(x.mean(dim=1)), attentions
        # return self.linear(x.mean(dim=1)), x


class Encoder(nn.Module):  # mikham
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        global global_var1
        for layer in self.layers:
            x, attntions = layer(x, c)
        return self.norm(x), attntions


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, c, top_k_indices, random_indices, mode):
        "Apply residual connection to any sublayer with the same size."
        if mode == 'attn':
            top_ks = torch.index_select(x, dim=1, index=top_k_indices)
            random_k = torch.index_select(x, dim=1, index=random_indices)
            top_k = torch.cat((top_ks, random_k), dim=1)
            multiheadedattn = sublayer(self.norm(x))
            return top_k + self.dropout(multiheadedattn[0]), multiheadedattn[1]
        elif mode == 'ff':
            return x + self.dropout(sublayer(self.norm(x)))

        # if mode =='attn':
        #     _, m_indices = torch.sort(c, 1, descending=True)
        #     top_k_indices = m_indices[:, 0:k // 2, :].squeeze()
        #     top_k_half = torch.index_select(x, dim=1, index=top_k_indices)
        #     random_k = torch.index_select(x, dim=1, index=random_indices)
        #     top_k = torch.cat((top_k_half, random_k), dim=1)
        #     return top_k + self.dropout(sublayer(self.norm(x)))
        # elif mode =='ff':
        #     return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout, k, random_patch_share):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.k = k
        self.random_patch_share = random_patch_share
        self.top_k_share = 1.0 - random_patch_share

    def forward(self, x, c):
        "Follow Figure 1 (left) for connections."
        _, m_indices = torch.sort(c, 1, descending=True)
        top_k_share_indices = m_indices[:, 0:math.ceil(self.k * self.top_k_share), :].squeeze()
        top_ks = torch.index_select(x, dim=1, index=top_k_share_indices)

        if top_k_share_indices.dim() == 0:  # If topk_share_indices is a Scalar tensor, convert it to 1-D tensor
            top_k_share_indices = top_k_share_indices.unsqueeze(0)

        remaining_indices = list(set(range(x.shape[1])) - set(top_k_share_indices.tolist()))
        randoms_share = min(
            int(self.k * self.random_patch_share),
            max(0, x.shape[1] - math.ceil(self.k * self.top_k_share))
        )  # if 500: min (250, max(0, num_patches - 250))
        random_indices = torch.from_numpy(
            np.random.choice(remaining_indices, randoms_share, replace=False)).cuda()
        random_k = torch.index_select(x, dim=1, index=random_indices)
        top_k = torch.cat((top_ks, random_k), dim=1)
        x, attentions = self.sublayer[0](x, lambda x: self.self_attn(x, top_k, x), c, top_k_share_indices,
                                         random_indices, 'attn')
        return self.sublayer[1](x, self.feed_forward, c, None, None, 'ff'), attentions

        # _, m_indices = torch.sort(c, 1, descending=True)
        # top_k_indices = m_indices[:, 0:self.k // 2, :].squeeze()
        # top_k_half = torch.index_select(x, dim=1, index=top_k_indices)
        # remaining_indices = list(set(range(self.k)) - set(top_k_indices.tolist()))
        # random_indices = torch.from_numpy(np.random.choice(remaining_indices, self.k // 2, replace=False)).cuda()
        # if self.training:
        #     self.random_indices = nn.Parameter(random_indices, requires_grad=False).cuda()
        # else:
        #     random_indices = self.random_indices
        # random_k = torch.index_select(x, dim=1, index=random_indices)
        # top_k = torch.cat((top_k_half, random_k), dim=1)
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, top_k, x), c, self.k, random_indices, 'attn')
        # return self.sublayer[1](x, self.feed_forward, c, self.k, random_indices, 'ff'), random_indices


def attention(query, key, value, dropout=None):  # mikham bi mask, input behesh bayad topk bashe
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.transpose(-2, -1), value), p_attn


class MultiHeadedAttention(nn.Module):  # ye joor be in bayad query, key matloob bedam

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Figure 2"
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(  # be in topk bedam
            query, key, value, dropout=self.dropout
        )
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x), self.attn


class PositionwiseFeedForward(nn.Module):  # mikham
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # print('d_ff:', print(d_ff))
        # raise Exception('d_ff')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A = self.b_classifier(feats, classes)

        return classes, prediction_bag, A
