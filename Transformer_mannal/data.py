import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


dict_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'

dict_x = {word: i for i, word in enumerate(dict_x.split(','))}
dict_y = {k.upper(): v for k, v in dict_x.items()}

dict_xr = dict_x.keys()
dict_yr = dict_y.keys()

def getData():
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
    ]
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()

    n = random.randint(30, 48)

    x = np.random.choice(words, size=n, replace=True, p=p)
    x = x.tolist()

    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)

    y = [f(i) for i in x]
    y = y + [y[-1]]
    y = y[::-1]

    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度     这种补pad的方式可太暴力了
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

class MyDataSet(Dataset):
    def __init__(self):
        super(MyDataSet, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, item):
        return getData()


loader = DataLoader(dataset=Dataset(),
                    batch_size=8,
                    drop_last=True,
                    shuffle=True,
                    collate_fn=None)


