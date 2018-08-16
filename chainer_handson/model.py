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



