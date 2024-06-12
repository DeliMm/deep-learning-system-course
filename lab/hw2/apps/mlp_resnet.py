import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    mlp_list = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        mlp_list.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    mlp_list.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*mlp_list)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    acc, loss, cnt = 0, 0, 0
    loss_f = nn.SoftmaxLoss()
    if opt:
        model.train()

        for (x, y) in dataloader:
           # import pdb; pdb.set_trace()\
            logits = model(x)
            loss_v = loss_f(logits, y)
            loss += loss_v.data * x.shape[0]
            cnt += x.shape[0]
            acc += np.sum(np.argmax(logits.numpy(), axis=-1) == y.numpy())
            opt.reset_grad()
            loss_v.backward()
            opt.step()
            
    else:
        model.eval()
        for (x, y) in dataloader:
            logits = model(x)
            loss_v = loss_f(logits, y)
            cnt += x.shape[0]
            acc += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy()) 
            loss += loss_v.data * x.shape[0]

    return np.array(1 - acc / cnt, dtype=np.float32), np.array(loss.numpy() / cnt, dtype=np.float32) 
    ### END YOUR SOLUTION

def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_ds = ndl.data.datasets.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), 
                                              os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    train_loader = ndl.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = ndl.data.datasets.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), 
                                             os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    test_loader = ndl.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err, train_loss =epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
