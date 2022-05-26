import torch
import torch.nn as nn

from data import dict_y, loader, dict_xr, dict_yr
from mask import mask_pad, mask_tril
from model import Transformer

model = Transformer()
epoch_num = 1
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

schedule = torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=3,
                                           gamma=0.5)

def predict(x):

    model.eval()
    mask_pad_x = mask_pad(x)

    # target = [1, 50]
    target = [dict_y["<SOS>"]] + [dict_y['<PAD>']] * 49
    # target = [1, 1, 50]
    target = torch.LongTensor(target).unsqueeze(0)

    x = model.embed_x(x)

    x = model.encoder(x, mask_pad_x)

    for i in range(49):
        y = target

        mask_tril_y = mask_tril(y)

        y = model.embed_y(y)

        y = model.decoder(x, y, mask_pad_x, mask_tril_y)

        out = model.fc_out(y)

        out = out[:, i, :]

        out = out.argmax(dim=1).detach()

        target[:, i+1] = out
    return target


for epoch in range(epoch_num):
    for i, (x, y) in enumerate(loader):
        # x = [8, 50]
        # y = [8, 51]

        # pred = [8, 50, 39]
        pred = model(x, y[:, :-1])

        pred = pred.reshape(-1, 39)

        y = y[:, 1:].reshape(-1)

        select = y != dict_y['<PAD>']
        pred = pred[select]
        y = y[select]

        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backword()
        optimizer.step()

        if i % 200 == 0:
            pred = pred.argmax(1)   # dim=1
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            lr = optimizer.param_groups[0]['lr']
            print(epoch, i, lr, loss.item(), accuracy)
    schedule.step()

