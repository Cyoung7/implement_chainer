# -*-coding:utf-8-*-
from chainer.datasets import mnist
from chainer import iterators
import chainer.functions as F
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
from chainer_handson.model import MLP
from chainer import serializers
import numpy as np
import matplotlib.pyplot as plt

train, test = mnist.get_mnist(withlabel=True, ndim=1)
# 数据预览
# x,t = train[0]
# plt.imshow(x.reshape(28,28),cmap='gray')
# plt.show()
# print('label:', t)

# create the dataset iterator
batch_size = 128
train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

gpu_id = 0
model = MLP()
if gpu_id >= 0:
    model.to_gpu(gpu_id)

optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

max_epoch = 10
while train_iter.epoch < max_epoch:
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch, gpu_id)
    pred_train = model(image_train)
    loss = F.softmax_cross_entropy(pred_train, target_train)

    model.cleargrads()
    loss.backward()
    optimizer.update()
    if train_iter.is_new_epoch:
        print('epoch:{:02d} train loss:{:.04f}'.
              format(train_iter.epoch, float(to_cpu(loss.data))), end='  ')
        test_losses = []
        test_accuracies = []
        while True:
            test_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch, gpu_id)
            pred_test = model(image_test)
            loss_test = F.softmax_cross_entropy(pred_test, target_test)
            test_losses.append(to_cpu(loss_test.data))

            test_acc = F.accuracy(pred_test, target_test)
            test_acc.to_cpu()
            test_accuracies.append(test_acc.data)

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                break
        print('val loss:{:.04f} val_accuracy:{:.04f}'.
              format(np.mean(test_losses), np.mean(test_accuracies)))

# save train model
serializers.save_npz('my_mnist.model', model)

print('hello')
