import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, variable
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer.optimizer_hooks import WeightDecay
import numpy as np
import matplotlib as plt
import os
from os.path import expanduser

# HYPER PARAM
BATCH_SIZE = 1

class Net(chainer.Chain):
	def __init__(self, n_channel=3, n_action=4):
		initializer = chainer.initializers.HeNormal()
		super(Net, self).__init__(
			conv1=L.Convolution2D(n_channel, 32, ksize=8, stride=4, nobias=False, initialW=initializer),
			conv2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False, initialW=initializer),
			conv3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, initialW=initializer),
			fc4=L.Linear(960, 512, initialW=initializer),
			fc5=L.Linear(512, n_action, initialW=np.zeros((n_action, 512), dtype=np.int32))
			)
	def __call__(self, x, test=False):
		s = chainer.Variable(x)
		h1 = F.relu(self.conv1(s))
		h2 = F.relu(self.conv2(h1))
		h3 = F.relu(self.conv3(h2))
		h4 = F.relu(self.fc4(h3))
		h = self.fc5(h4)
		return h

class deep_learning:
	def __init__(self, n_channel=3, n_action=3):
		self.net = Net(n_channel, n_action)
		self.optimizer = chainer.optimizers.Adam(eps=1e-2)
		self.optimizer.setup(self.net)
		self.optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
		self.n_action = n_action
		self.phi = lambda x: x.astype(np.float32, copy=False)
		self.count = 0
		self.accuracy = 0
		self.results_train = {}
		self.results_train['loss'], self.results_train['accuracy'] = [], []
		self.loss_list = []
		self.acc_list = []

	def act_and_trains(self, imgobj, correct_action):
			x = [self.phi(s) for s in [imgobj]]
			t = np.array([correct_action], np.int32)
			dataset = TupleDataset(x, t)
			train_iter = SerialIterator(dataset, batch_size = BATCH_SIZE, repeat=True, shuffle=False)
			train_batch  = train_iter.next()
			x_train, t_train = chainer.dataset.concat_examples(train_batch, -1)

			y_train = self.net(x_train)

			loss_train = F.softmax_cross_entropy(y_train, t_train)
			acc_train = F.accuracy(y_train, t_train)

			self.loss_list.append(loss_train.array)
			self.acc_list.append(acc_train.array)

			self.net.cleargrads()
			loss_train.backward()
			self.optimizer.update()
			
			self.count += 1

			self.results_train['loss'] .append(loss_train.array)
			self.results_train['accuracy'] .append(acc_train.array)

			action = np.argmax(y_train.array)
			self.accuracy = np.mean(self.acc_list)
#			print('iteration: {}, acc (train): {:.4f}, action: {}'.format(self.count, self.accuracy, action))

			return action

	def act(self, imgobj):
			x = [self.phi(s) for s in [imgobj]]
			x_test = chainer.dataset.concat_examples(x, -1)

			with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
				action_value = self.net(x_test)
				action = np.argmax(action_value.array)
			return action



	def result(self):
			accuracy = self.accuracy
			return accuracy

if __name__ == '__main__':
        dl = deep_learning()
