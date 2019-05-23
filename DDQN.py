#!/usr/bin/env python
from __future__ import print_function # 新版本特性

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque # 双端队列

# 参数
GAME = 'bird' # 游戏名称
ACTIONS = 2 # 动作种类 上or下
GAMMA = 0.99 # Q-learning 衰减率α
OBSERVE = 100. # 经验池的样本数
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # 结束探索时候的选择动作的ε概率
INITIAL_EPSILON = 0.01 # 开始探索时候的选择动作的ε概率
REPLAY_MEMORY = 50000 # 经验池的最大内存
BATCH = 32 # 随机抽样的样本数
FRAME_PER_ACTION = 1
UPDATE_TIME = 100 #更新目标网络


class DDQN:

	def __init__(self):
		# 初始化经验池
		self.memory = deque()
		# 初始化步数 检测模型保存和EPSILION的改变
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		
		# 初始化当前Q网络
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createNetwork()
		# 初始化目标Q网络
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createNetwork()
		#将当前Q网络赋值给目标Q网络 tf.assign为赋值操作
		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
		#初始化损失函数
		self.createTrainingMethod()
		
		# 保存和加载网络模型
		# TensorFlow采用Saver来保存。一般在Session()建立之前，通过tf.train.Saver()获取Saver实例
		self.saver = tf.train.Saver()
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.initialize_all_variables())
		#如果检查点存在就载入已经有的模型
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
		
		
	# 初始化当前状态
	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

    # 构建CNN卷积神经网络
    # 权重 tf.truncated_normal(shape, mean, stddev):
    #      shape表示生成张量的维度，mean是均值，stddev是标准差 一个截断的产生正太分布的函数 
    #      TensorFlow的世界里，变量的定义和初始化是分开的 tf.Variable(initializer,name),initializer是初始化参数，name是可自定义的变量名称
	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.01) 
		return tf.Variable(initial)

    # 偏置 TensorFlow创建常量tf.constant
	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	# 卷积 tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
	#      input -- 卷积输入图像 Tensor [batch, in_height, in_width, in_channels] [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数] 
	#      filter -- 卷积核 Tensor [filter_height, filter_width, in_channels, out_channels] [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] 
	#      strides -- 卷积时在图像每一维的步长 步长不为1的情况，文档里说了对于图片，因为只有两维，通常strides取[1，stride，stride，1]
	#	   padding --  "SAME","VALID" SAME： 输出大小等于输入大小除以步长 VALID: 输出大小等于输入大小减去滤波器大小加上1，最后再除以步长 向上取整
	def conv2d(self, x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	# 池化 tf.nn.max_pool(value, ksize, strides, padding, name=None)
	#      输入 [batch, height, width, channels] 
	#	   池化窗口大小 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
	#      步长 和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
	#	   填充 "SAME","VALID"
	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	# 构建CNN模型 inputState QValue
	def createNetwork(self):
		# 第一层卷积 卷积核 8*8*4*32
		W_conv1 = self.weight_variable([8, 8, 4, 32])
		b_conv1 = self.bias_variable([32])
		# 第二层卷积 卷积核 4*4*32*64
		W_conv2 = self.weight_variable([4, 4, 32, 64])
		b_conv2 = self.bias_variable([64])
		# 第三层卷积 卷积核 3*3*64*64
		W_conv3 = self.weight_variable([3, 3, 64, 64])
		b_conv3 = self.bias_variable([64])
		# 第一层全连接 1600 - 512
		W_fc1 = self.weight_variable([1600, 512])
		b_fc1 = self.bias_variable([512])
		# 第二层全连接 512 - 2
		W_fc2 = self.weight_variable([512, ACTIONS])
		b_fc2 = self.bias_variable([ACTIONS])
		
		# 输入层
		stateInput = tf.placeholder("float", [None, 80, 80, 4])

		# 第一层隐藏层+池化层 tf.nn.relu激活函数  80*80*4 ->  20*20*32  80/4 = 20
		h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1) #  80*80*4 ->  20*20*32  80/4 = 20
		h_pool1 = self.max_pool_2x2(h_conv1) # 20*20*32 -> 10*10*32  20/2 = 10
		# 第二层隐藏层（这里只用了一层池化层）
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2) # 10*10*32 -> 5*5*64  10/2 = 5
		# h_pool2 = max_pool_2x2(h_conv2)
		# 第三层隐藏层
		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3) # 5*5*64 -> 5*5*64  5/1 = 5
		# h_pool3 = max_pool_2x2(h_conv3)
		# Reshape
		#h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
		h_conv3_flat = tf.reshape(h_conv3, [-1, 1600]) # 5*5*64 = 1600 n*1600 --1600
		# 全连接层
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)  # 1600*512 -- 512
		# 输出层
		# readout layer 动作的Q值
		QValue = tf.matmul(h_fc1, W_fc2) + b_fc2 # 512*2 -- 2
		
		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2
		
	#赋值目标Q网络
	def copyTargetQNetwork(self):
		self.sess.run(self.copyTargetQNetworkOperation)
		
		# 损失函数
	def createTrainingMethod(self):
		# 这里的actionInput表示输出的动作，即强化学习模型中的Action，yInput表示标签值，Q_action表示模型输出与actionInput相乘后，在一维求和，损失函数对标签值与输出值的差进行平方
		self.actionInput = tf.placeholder("float", [None, ACTIONS]) # 输出动作
		self.yInput = tf.placeholder("float", [None]) # 标签
		Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
		# train_step表示对损失函数进行Adam优化。
		self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
		
		
	# 训练网络
	def trainNetwork(self):  # 图片输入 输出层 全连接层 tf训练
    
		# 梯度下降
		# 获取最训练数据
		minibatch = random.sample(self.memory, BATCH)
		state_batch  = [d[0] for d in minibatch]
		action_batch = [d[1] for d in minibatch]
		reward_batch = [d[2] for d in minibatch]
		next_state_batch = [d[3] for d in minibatch]
		# 计算标签值
		y_batch = []
		current_Q_batch = self.QValue.eval(feed_dict = {self.stateInput : next_state_batch})
		max_action_next = np.argmax(current_Q_batch, axis = 1)
		QValue_batch = self.QValueT.eval(feed_dict = {self.stateInputT : next_state_batch})
		Selected_q_next = QValue_batch[range(len(max_action_next)), max_action_next]
		for i in range(0, BATCH):
			terminal = minibatch[i][4]
			# if terminal, only equals reward
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				
				y_batch.append(reward_batch[i] + GAMMA * Selected_q_next[i])
				
		# perform gradient step
		self.train_step.run(feed_dict = {
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})
		
		# save progress every 10000 iterations
		if self.timeStep % 10000 == 0:
			self.saver.save(self.sess, 'saved_networks/' + GAME + '-dqn', global_step = self.timeStep)
		#更新目标Q网络
		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

		#新的观察状态进入经验池
	def setPerception(self,nextObservation,action,reward,terminal):
		newState = np.append(self.currentState[:,:,1:], nextObservation, axis = 2)
		self.memory.append((self.currentState, action, reward, newState, terminal))
		#防止经验池内存占用过大
		if len(self.memory) > REPLAY_MEMORY:
			self.memory.popleft()
		# 开始训练网络
		if self.timeStep > OBSERVE:
			self.trainNetwork()
		
		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print ("TIMESTEP", self.timeStep, "/ STATE", state, \
			"/ EPSILON", self.epsilon)


		self.currentState = newState
		self.timeStep = self.timeStep + 1
	
	def getAction(self):
		# 根据ε 概率选择一个Action
		QValue = self.QValue.eval(feed_dict={self.stateInput : [self.currentState]})[0]
		action = np.zeros([ACTIONS])
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				print("----------Random Action----------")
				action_index = random.randrange(ACTIONS)
				action[action_index] = 1
			else:
				print("----------QNetwork Action----------")
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1 # do nothing
		
		# 缩减ε
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		return action
		
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))
		
		
def playGame():
	#初始化DDQN
	brain = DDQN()
	# 打开游戏已经仿真通信器
	flappyBird = game.GameState()
	# 开始游戏
	# 获得出事状态
	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal = flappyBird.frame_step(action0)
	# 首先将图像转换为80*80，然后进行灰度化
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	# 对灰度图像二值化
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)

	# 开始游戏
	while 1!= 0:
		action = brain.getAction()
		nextObservation,reward,terminal = flappyBird.frame_step(action)
		nextObservation = preprocess(nextObservation)
		brain.setPerception(nextObservation,action,reward,terminal)
	

def main():
    playGame()

if __name__ == "__main__":
    main()
	
	
	
