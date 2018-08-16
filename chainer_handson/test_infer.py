# -*-coding:utf-8-*-
from chainer_handson.model import MLP
from chainer.datasets import mnist
from chainer import iterators
from chainer.dataset import concat_examples
from chainer import serializers
from chainer.backends.cuda import to_gpu
import numpy as np
from matplotlib import pyplot as plt

infer_model = MLP()
serializers.load_npz('mnist_result/model_epoch-10', infer_model)
gpu_id = 0
if gpu_id >= 0:
    infer_model.to_gpu(gpu_id)

# ndarray
train, test = mnist.get_mnist(withlabel=False, ndim=1)
# train_iter = iterators.SerialIterator(train,batch_size=128,repeat=False,shuffle=False)
# test_iter = iterators.SerialIterator(test,batch_size=128,repeat=False,shuffle=False)
#
# train_batch = test_iter.next()
# train_image = concat_examples(train_batch,device=0)

x = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()
print('shape:', x.shape, end='->')
x = np.expand_dims(x, axis=0)
print('shape:', x.shape)

if gpu_id >= 0:
    x = to_gpu(x, 0)

y = infer_model(x)
pred_label = y.data.argmax(axis=1)
print('predicted label:', pred_label[0])
print('hello')
