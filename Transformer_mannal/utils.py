import torch
import torch.nn as nn
import math


def attention(Q, K, V, mask):
    # b句话， 每句话50个词， 每个词32为词嵌入， 4个头
    # Q, K, V = [b, 4, 50, 8]
    # score = [b, 4, 50, 50]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score /= 8 ** 0.5

    # mask遮盖，mask是True的地方都被替换成-inf，这样在计算softmax时，-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    # score = [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # [b, 4, 50, 8] --> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)

    return score

class MultiHead(nn.Module):
    def __init__(self):
        super(MultiHead, self).__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = nn.Linear(32, 32)

        self.norm = nn.LayerNorm(normalized_shape=32,
                                 elementwise_affine=True)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # [b, 50, 32]
        b = Q.shape[0]

        clone_Q = Q.clone()

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        score = attention(Q, K, V, mask)

        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score


class PositionEmbedding(nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()

        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** ((2*i) / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        # [50, 32] --> [1, 50, 32]
        pe = torch.unsqueeze(pe, dim=0)

        self.register_buffer('pe', pe)

        self.embed = nn.Embedding(39, 32)

        self.embed.weight.data.normal_(0, 0.1)  # 均值 方差

    def forward(self, x):
        # [8, 50] --> [8, 50, 32]
        embed = self.embed(x)
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed += self.pe
        return embed

class FullyConnectedOutput(nn.Module):
    def __init__(self):
        super(FullyConnectedOutput, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(p=0.1)
        )

        self.norm = nn.LayerNorm(normalized_shape=32,
                                 elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()

        x = self.norm(x)
        out = self.fc(x)
        out = clone_x + out
        return out





