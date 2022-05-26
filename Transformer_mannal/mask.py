import torch

from data import dict_x, dict_y


def mask_pad(data):
    """
    正常mask
    """
    # b句话每句话50个字母
    # data = [b, 50]
    # 判断每个词是不是<PAD>
    # mask = [b, 50]
    mask = data == dict_x['<PAD>']

    # [b, 50] --> [b, 1, 1, 50]
    mask = mask.reshape[-1, 1, 1, 50]
    mask = mask.expand(-1, 1, 50, 50)

    # return mask = [b, 1, 50, 50]
    return mask

def mask_tril(data):
    """
    上三角mask
    [[0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]
     """
    # tril 会生成一个下三角矩阵 需要提供一个初始矩阵
    # [1, 50, 50]
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))  # 这是一句话的mask

    # 判断y中的每个词是不是pad pad不可见

    mask = data == dict_y["<PAD>"]
    # [b, 1, 50]
    mask = mask.unsqueeze(1).long()

    # mask = [b, 50, 50]
    mask = mask + tril

    mask = mask > 0

    # 转bool型 [b, 1, 50, 50]
    mask = mask.unsqueeze(dim=1)

    return mask




