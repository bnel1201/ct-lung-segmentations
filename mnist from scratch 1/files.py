from fastai.vision.all import *

def load_train_and_valid(path, ntrain, nval):
    train_x = torch.stack([torch.stack([tensor(Image.open(f))
                            for f in (path/f'training/{i}').ls()[:ntrain//10]])
                                   for i in range(10)]).view(ntrain, 28*28)
    train_x = train_x.float()/255
    train_y = torch.stack(
    [tensor([i]*(ntrain//10)) for i in range(10)]
                        ).view(ntrain)
    
    valid_x = torch.stack([torch.stack([tensor(Image.open(f))
                            for f in (path/f'training/{i}').ls()[-nval//10:]])
                                   for i in range(10)]).view(nval, 28*28)
    valid_x = torch.stack([torch.stack([tensor(Image.open(f))
                            for f in (path/f'training/{i}').ls()[-nval//10:]])
                                   for i in range(10)]).view(nval, 28*28)
    valid_x = valid_x.float()/255
    valid_y = torch.stack(
    [tensor([i]*(nval//10)) for i in range(10)]
                        ).view(nval)
    return (train_x, train_y), (valid_x, valid_y)


def make_datasets(train_x, train_y, valid_x, valid_y):
    return list(zip(train_x, train_y)), list(zip(valid_x, valid_y))


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = torch.argmax(xb, dim=1) == yb
    return correct.float().mean()