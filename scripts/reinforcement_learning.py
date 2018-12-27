import chainerrl
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os
from os.path import expanduser
import csv

class QFunction(chainer.Chain):
	def __init__(self, n_history=3, n_action=4):
		initializer = chainer.initializers.HeNormal()
		super(QFunction, self).__init__(
			conv1=L.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False, initialW=initializer),
			conv2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False, initialW=initializer),
			conv3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, initialW=initializer),
			conv4=L.Linear(960, 512, initialW=initializer),
			fc5=L.Linear(512, n_action, initialW=np.zeros((n_action, 512), dtype=np.float32))
		)

	def __call__(self, x, test=False):
		global h5

		s = chainer.Variable(x)
		h1 = F.relu(self.conv1(s))
		h2 = F.relu(self.conv2(h1))
		h3 = F.relu(self.conv3(h2))
		h4 = F.relu(self.conv4(h3))
		h5 = self.fc5(h4)
		h = chainerrl.action_value.DiscreteActionValue(h5)
		return h

class reinforcement_learning:
	def __init__(self, n_history=3, n_action=4):
		self.q_func = QFunction(n_history, n_action)
		try:
			self.q_func.to_gpu()
		except:
			print("No GPU")
		self.optimizer = chainer.optimizers.Adam(eps=1e-2)
		self.optimizer.setup(self.q_func)
		self.gamma = 0.95
		self.n_action = n_action
		self.explorer = chainerrl.explorers.ConstantEpsilonGreedy(
			epsilon=0.1, random_action_func=self.action_space_sample)
		self.replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 4)
		self.phi = lambda x: x.astype(np.float32, copy=False)
		self.agent = chainerrl.agents.DoubleDQN(
			self.q_func, self.optimizer, self.replay_buffer, self.gamma, self.explorer,
			minibatch_size=4, replay_start_size=100, update_interval=1,
			target_update_interval=100, phi=self.phi)

		home = expanduser("~")
		if os.path.isdir(home + '/agent'):
			self.agent.load('agent')
			print('agent LOADED!!')

	def act_and_trains(self, obs, reward):
		self.action = self.agent.act_and_train(obs, reward)
		return self.action
	def stop_episode_and_train(self, obs, reward, done):
		self.agent.stop_episode_and_train(obs, reward, done)
		print('\x1b[6;30;42m' + 'Last step in this episode' + '\x1b[0m')
	def act(self, obs):
		self.action = self.agent.act(obs)
		action_prob = F.softmax(h5)[0]
		action_prob = str(action_prob)
		action_prob = action_prob.replace("variable([", "")
		action_prob = action_prob.replace("])", "")
		action_prob = map(float, action_prob.split())
		return self.action, action_prob

	def save_agent(self):
		self.agent.save('agent')
		print("agent SAVED!!")

	def action_space_sample(self):
		return np.random.randint(1,self.n_action)

if __name__ == '__main__':
	rl = reinforcement_learning()
