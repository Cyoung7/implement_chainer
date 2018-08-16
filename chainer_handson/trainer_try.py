# -*-coding:utf-8-*-
import chainer
from chainer.datasets import mnist
from chainer import iterators
from chainer import optimizers
from chainer import training
import chainer.links as L
from chainer.training import extensions
from chainer_handson.model import MLP

train, test = mnist.get_mnist(withlabel=True, ndim=1)
train_iter = iterators.SerialIterator(train, batch_size=128, repeat=True, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=128, repeat=False, shuffle=False)

gpu_id = 0
model = MLP()

max_epoch = 10
model = L.Classifier(model)
if gpu_id >= 0:
    model = model.to_gpu(gpu_id)

optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                       'validation/main/loss', 'validation/main/accuracy',
                                       'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                     x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                     x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor,
                                          filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()
