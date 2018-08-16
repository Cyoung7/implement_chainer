# -*-coding:utf-8-*-
import chainer
import chainer.links as L
import chainer.functions as F


class MLP(chainer.Chain):
    def __init__(self, hidden_unit=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, hidden_unit)
            self.l2 = L.Linear(None, hidden_unit)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class LeNet(chainer.Chain):
    def __init__(self, n_out=10):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, 3, 1)
            self.conv2 = L.Convolution2D(32, 64, 3, 3, 1)
            self.conv3 = L.Convolution2D(64, 128, 3, 3, 1)
            self.l1 = L.Linear(None, 1000)
            self.l2 = L.Linear(1000, 10)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        # Linear 会自动reshape成1维
        h4 = F.relu(self.l1(h3))
        h5 = self.l2(h4)
        return h5
