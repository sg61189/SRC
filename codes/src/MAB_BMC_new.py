import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
from scipy import interpolate
import tracemalloc
# from memory_profiler import memory_usage
from abcBMCUtil import *

DEBUG = True
DEBUG = False
OPT = True
T = 60 
TIMEOUT = 3600
SC = 2
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
# DIFF = 2 # function of depth, time, memory
DIFF = 3 # function number of frames/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
FIXED = False
FIXED = True
PDR = False
MAX_FRAME = 1000
MAX_CLAUSE = 1e9
MAX_TIME = 3600
MAX_MEM = 2000

time_outs = {}
Actions = ['bmc2', 'bmc3', 'bmc3rs', 'bmc3j', 'bmc3rg', 'bmcru', 'bmc3r', 'pdr']

class bandit:

	def __init__(self, k, iters, alpha, reward=0, fname = ''):
		# Number of arms
		self.k = k
		# Number of iterations
		self.iters = iters
		# Step count
		self.n = 0
		# Step count for each arm
		self.k_n = np.zeros(k)
		# Total mean reward
		self.mean_reward = 0
		self.reward = np.zeros(iters)
		# Mean reward for each arm
		self.k_reward = np.zeros(k)

		for i in range(self.k):
			self.k_reward[i] = reward

		# recency constant
		self.alpha = alpha
		##
		self.fname =  fname
		self.timeout = np.zeros(iters)
		self.timeout[0] = T

		self.engine_res = [{} for i in range(k)]

		# current state of the run; eg - bmc depth
		self.states = 0

		self.explore = False

	def get_reward(self, a, t1 = -1):

		# get starting depth
		# sd = 0
		# if self.n  == 0:
		sd = int(self.states) #[a])

		fname = os.path.join(PATH, self.fname)
		#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
		ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))
		if DEBUG or self.n == 0:
			print(fname, ofname)

		st = ''

		if t1  == -1:
			t = T
		else:
			t = int(t1)

		if self.n == 0:
			simplify(fname, ofname)
			print('Simplified model', ofname)

		sm = None
		asrt = -1
		ar_tab = {}
		if a == 0:    #ABC bmc2
			asrt, sm, ar_tab = bmc2(ofname, sd, t=t)
		elif a == 1: #ABC bmc3
			asrt, sm, ar_tab = bmc3(ofname, sd, t=t)
		elif a == 2: #ABC bmc3rs
			asrt, sm, ar_tab = bmc3rs(ofname, sd, t=t)
		elif a == 3: #ABC bmc3j
			asrt, sm, ar_tab = bmc3j(ofname, sd, t=t)
		elif a == 4: #ABC bmc3g
			asrt, sm, ar_tab = bmc3rg(ofname, sd, t=t)
		elif a == 5: #ABC bmc3u
			asrt, sm, ar_tab = bmc3ru(ofname, sd, t=t)
		elif a == 6: #ABC bmc3r
			asrt, sm, ar_tab = bmc3r(ofname, sd, t=t)
		elif a == 7: #ABC pdr
			asrt, sm, ar_tab = pdr(ofname, t)

		ar_tab_old = self.engine_res[a]
		for ky in ar_tab.keys():
			ar_tab_old.update({ky:ar_tab[ky]})

		self.engine_res[a] = ar_tab_old

		if sm is not None:
			# print(sm)
			reward = 0
			if DIFF == 0 :
				reward = (sm.frame - sd)
			elif DIFF == 1:
				reward = sm.frame
			elif DIFF == 2:
				pen = t - sm.tt
				reward =  np.exp(2*sm.frame/MAX_FRAME - 0.5*sm.mem/MAX_MEM - 0.6*sm.tt/MAX_TIME)
				# + 0.1*sm.cla/MAX_CLAUSE 
				#np.exp(0.5*sm.frame/MAX_FRAME + 0.1*sm.cla/MAX_CLAUSE - 0.3*sm.to/MAX_TIME) #-  0.1*pen/MAX_TIME)
				if asrt > 0:
					reward =  np.exp(1 - 0.4*sm.frame/MAX_FRAME - 0.6*sm.tt/MAX_TIME)
			else:
				#reward = sm.cla/(10000 * sm.to) if sm.to > 0 else sm.to

				pen = t - sm.tt
				reward = 0
				cn = 0
				for ky in ar_tab_old.keys():
					tm = ar_tab_old[ky].to
					reward += 2*np.exp(-1*tm/(1+ky))
					cn += 1
				reward = (reward)/cn #(reward + np.exp(-pen/MAX_TIME))/cn
		else:
			sm =  abc_result(frame=sd, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = t)
			if asrt > 0:
				reward = asrt
			else:
				reward = -0.5 * np.exp(t/MAX_TIME) #np.log(t)
		print(reward, sm)

		return reward, sm, ar_tab_old


	def run(self):
		totalTime = 0
		seq = []
		sm = None
		# next_to = -1
		# next_time = {}
		# # for i in self.iters:
		# #     next_time[i] = -1
		# for a in range(self.k):
		# 	time_outs.update({a:next_time})
		# frames = {}
		repeat_count = 2*self.k
		count = 0
		best_sd = 0
		flag = 0
		once = 0
		twice = 0
		ocount = 0
		ending = 0
		critical = 0

		conf_begin_phase = 0
		max_conf = 0
		a = 0
		for i in range(self.iters):
			if int(TIMEOUT - totalTime) <= 0:
				print('BMC-depth reached ', self.states, 'totalTime', totalTime)
				print('Stopping iteration')
				break

			#self.explore = False
			a = self.pull(a, count)
			if i < repeat_count or flag == 1:
				a = i%(self.k)
				# self.explore = True
				# a = self.pull(a, 1)
				#
				#


			# if a in time_outs:
			#     nextt = time_outs[a]
			#     if self.states in nextt:
			#         next_to = nextt[self.states]

			if i < repeat_count:
				self.timeout[i] = T
			else:
				if i > 1 and flag == 1:
					self.timeout[i] = self.timeout[i-1]
					ocount += 1
				else:
					# if FIXED:
					# if next_to == -1:

					if sm and sm.to == -1 and count > 2:
						next_timeout = self.timeout[i-1] * SC
						if self.timeout[i] >= 120:
							critical = 1
						if ending :
						#if self.timeout[i-1] > next_timeout > TIMEOUT - (totalTime + self.timeout[i]):
						#	if count > self.k-1:
							print('BMC-depth reached ', self.states, 'totalTime', totalTime)
							print('Stopping iteration - condition with timeout ', next_timeout)
							break
							# else:
							# 	self.timeout[i] = next_timeout #min(self.timeout[i-1] * SC, 480)
							# 	count = 0
						else:
							self.timeout[i] = next_timeout #min(self.timeout[i-1] * SC, 480)
							count = 0
					else:
						self.timeout[i] = self.timeout[i-1]
						count = count + 1
					# else:
					# try:
					# 	if sm and sm.to == -1 and count > 2:
					# 		self.timeout[i] = max(math.ceil(next_to), self.timeout[i-1]*SC, 480)
					# 		count = 0
					# 	else:
					# 		self.timeout[i] = max(math.ceil(next_to), self.timeout[i-1])
					# 		count = count + 1

					# except OverflowError:
					# 	self.timeout[i] = self.timeout[i-1]
					# except ValueError:
					# 	self.timeout[i] = self.timeout[i-1]
			  
			print('Calculating time out', self.timeout[i], 'total till now', totalTime)

			if self.timeout[i] > TIMEOUT - (totalTime + self.timeout[i]):
				self.timeout[i] =  (TIMEOUT - totalTime) #self.timeout[i] +
				ending = 1
			else:
				self.timeout[i] = min(self.timeout[i], TIMEOUT - totalTime)

			print('Next time out', TIMEOUT, self.timeout[i], 'for chosen action', a, Actions[a], 'ocount', ocount,  'flag', flag, 'once', once, 'ending', ending)

			if self.timeout[i] < 1.0:
				print('BMC-depth reached ', self.states, 'totalTime', totalTime)
				print('Stopping iteration - condition with timeout < ', 1.0)
				break

			a, reward, sm, ar_tab = self.update_policy(a, self.timeout[i])

			# fragmentation
			tt = math.ceil(sm.tt) #if sm.asrt > 0  else self.timeout[i]
		
			if sm and reward > 0:
				print(i, 'sm', 'conf', sm.conf, 'cla', sm.cla, max(2*conf_begin_phase, 1e5), 'once', once, 'flag', flag, 'iter', (i+1)%self.k)
				ss = (Actions[a], tt, reward, totalTime, self.timeout[i], self.states)
				if (i < repeat_count ) or (flag == 1) :
					sd = sm.frame+1 if sm.frame > 0 else sm.frame
					best_sd = max(best_sd, sd)
					if (i < repeat_count):
						max_conf = max(max_conf, sm.conf)
					print('------ exploring')
					if (i < repeat_count and i%self.k == self.k-1) or (flag == 1 and ocount == self.k-1) :
						# if flag == 1:
						# 	print('------ Stopped exploring beginning')
						# elif i < 2*self.k:
						print('------ at the end of exploration')
						# sd = sm.frame+1 if sm.frame > 0 else sm.frame
						# best_sd = max(best_sd, sd)
						self.states = best_sd
						ss = (Actions[a], tt, reward, totalTime, self.timeout[i], self.states)
						seq.append(ss)
						totalTime += tt
						if (i < repeat_count and i%self.k == self.k-1):
							conf_begin_phase = max_conf

				elif flag == 0 and i >= repeat_count:# and i%self.k == self.k-1 				
					print('------ no exploration')
					sd = sm.frame+1 if sm.frame > 0 else sm.frame  
					if a == 0:
						sd = sm.frame 
					self.states = sd
					ss = (Actions[a], tt, reward, totalTime, self.timeout[i], self.states)
					seq.append(ss)
					totalTime += tt
					
				if i > repeat_count and (sm.conf > max(2*conf_begin_phase, 1e5) or critical == 1) and once == 0:
					if flag == 0 and ocount == 0: #(i+1)%self.k == 0 :
						flag = 1
						print('------ Started exploring  once')
					if flag == 1 and ocount == self.k-1: #(i+1)%self.k == self.k-1:
						flag = 0
						once = 1
						#ocount = 0
						print('----- Stopped exploring once')

				# if i > 2*self.k and sm.conf > 6e5 and once == 1 and twice == 0:
				# 	if flag == 0 and (i+1)%self.k == 0 :
				# 		flag = 1
				# 		print('------ Started exploring twice')
				# 	if flag == 1 and (i+1)%self.k == self.k-1:
				# 		flag = 0
				# 		twice = 1
				# 		print('----- Stopped exploring twice')
			else:
				ss = (Actions[a], -1, reward, -1, self.timeout[i], self.states)
			
			self.reward[i] = self.mean_reward

			# if sm and sm.to > 0 and reward > 0:
			#     totalTime += tt
				
			print('#### iter ', i, Actions[a], 'time taken', tt, self.timeout[i], 'totalTime', totalTime, 'ss', ss, sm)

			# self.timeout[i] = T
			if sm and sm.asrt > 0:
				print('Output asserted at frame ', sm.asrt, 'tt', tt, 'totalTime', totalTime)
				print('Stopping iteration')
				break

			# if a in frames:
			# 	ftrain, ttrain, ttrain1 = frames[a]
			# else:
			# 	ftrain, ttrain,ttrain1 = [], [], []
			# for frm in ar_tab.keys():
			# 	sm1 = ar_tab[frm]
			# 	if sm1.conf > 0:
			# 		ftrain.append(sm1.frame)
			# 		ttrain.append(sm1.cla)
			# 		ttrain1.append(sm1.tt)
			# frames.update({a:(ftrain, ttrain, ttrain1)})
			# if len(frames) > 10:
			#     ftrain, ttrain = frames[-11:], time_outs[-11:]
			# else:
				# for a in range(self.k):
					# if a in frames:
					#     ftrain, ttrain = frames[a]
					# else:
					#     ftrain, ttrain = [], []
				# print('Training for action', a, len(ftrain), len(ttrain))
				# if len(ftrain) > 0:
				# 	# predict number of clauses then predict timeout for next frame
				# 	fcla = interpolate.interp1d(ftrain, ttrain, fill_value = 'extrapolate')
				# 	fto = interpolate.interp1d(ttrain, ttrain1, fill_value = 'extrapolate')
				# 	new_frames = np.arange(ftrain[-1]+1, ftrain[-1]+2, 1)
				# 	new_cla = fcla(new_frames)
				# 	new_to = fto(new_cla)
				# 	next_to = np.sum(new_to)
				# 	next_time.update({new_frames[0]: new_to[0]})
				# 	# print('predicted time out for next frame', new_frames, new_to)
				# # else:
				# #     next_to = -1
				# time_outs.update({a:next_time})
				# print('next time out', next_to)

		while i < self.iters:
			self.reward[i] = self.mean_reward
			i += 1
		print(i, 'BMC-depth reached ', sm.frame if sm else 0, 'totalTime', totalTime)
		print(i, 'Sequence of actions and BMC depth:', seq)
		return (sm.frame, sm.asrt, totalTime, seq)
			
	def reset(self, reward):
		# Resets results while keeping settings
		self.n = 0
		self.k_n = np.zeros(self.k)
		self.mean_reward = 0
		self.reward = np.zeros(self.iters)
		self.k_reward = np.zeros(self.k)
		for i in range(self.k):
			self.k_reward[i] = reward
		self.timeout = np.zeros(self.iters)


class eps_bandit(bandit):
	'''
	epsilon-greedy k-bandit problem
	
	Inputs
	=====================================================
	k: number of arms (int)
	eps: probability of random action 0 < eps < 1 (float)
	iters: number of steps (int)
	mu: set the average rewards for each of the k-arms.
		Set to "random" for the rewards to be selected from
		a normal distribution with mean = 0. 
		Set to "sequence" for the means to be ordered from 
		0 to k-1.
		Pass a list or array of length = k for user-defined
		values.
	'''
	
	def __init__(self, k, eps, iters, alpha = 1, reward = 0,fname = ''):
		bandit.__init__(self, k, iters, alpha, reward, fname)
		# Search probability
		self.eps = eps
		
	def pull(self, a1, count = 0):
		# Generate random number
		# np.random.seed(time.time())
		p = np.random.rand()
		if self.eps == 0 and self.n == 0:
			ap = [1,2,4]
			a = np.random.choice(ap)

			print('selection - cond1',a, p, self.n, self.eps, a1, count, self.explore)
		if count > 0:
			ap = []
			for i in range(self.k):
				if not(i == a1):
					ap.append(i)
			a = np.random.choice(ap)

			print('selection - cond2',a, p, self.n, self.eps, a1, count, self.explore)
		elif self.explore and p < self.eps :
			# Randomly select an action
			a = np.random.choice(self.k)
			# if count > 0:
			# 	ap = []
			# 	for i in range(self.k):
			# 		if not(i == a1):
			# 			ap.append(i)
			# 	a = np.random.choice(ap)

			print('selection - cond3',a, p, self.n, self.eps, a1, count, self.explore)
		else:
			# Take greedy action
			a = np.argmax(self.k_reward)

			print('selection - cond4',a, p, self.n, self.eps, a1, count, self.explore)

		return a

	def update_policy(self, a, t):
		# Execute the action and calculate the reward
		# mem_use, tm = memory_usage(self.get_reward(a))
		reward, sm, ar_tab = self.get_reward(a, t)
		
		# Update counts
		self.n += 1
		self.k_n[a] += 1
		
		# Update total
		self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
		
		# Update results for a_k
		if self.alpha == 1:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
		else:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) * self.alpha

		# if a >0:
		print('Action {0} reward {1}, All Reward {2}'.format(a, reward, self.k_reward))
		return a, reward, sm, ar_tab
		
   
class eps_decay_bandit(bandit):
	'''
	epsilon-decay k-bandit problem
	
	Inputs
	=====================================================
	k: number of arms (int)
	iters: number of steps (int)
	mu: set the average rewards for each of the k-arms.
		Set to "random" for the rewards to be selected from
		a normal distribution with mean = 0. 
		Set to "sequence" for the means to be ordered from 
		0 to k-1.
		Pass a list or array of length = k for user-defined
		values.
	'''
	
	def __init__(self, k, iters, alpha = 1, reward = 0, fname = ''):
		bandit.__init__(self, k, iters, alpha, reward, fname)
	   
	def pull(self, a1, count = 0):
		# Generate random number
		# np.random.seed(self.n)
		p = np.random.rand()
		if self.n == 0:
			ap = [1,2,4]
			a = np.random.choice(ap)
		if count > 0:
			ap = []
			for i in range(self.k):
				if not(i == a1):
					ap.append(i)
			a = np.random.choice(ap)
		elif p < 1 / (1 + self.n / self.k):
			# Randomly select an action
			a = np.random.choice(self.k)
		else:
			# Take greedy action
			a = np.argmax(self.k_reward)
		return a

	def update_policy(self, a, t):
		# Execute the action and calculate the reward
		# mem_use, tm = memory_usage(self.get_reward(a))
		reward, sm, ar_tab = self.get_reward(a, t)
		
		# Update counts
		self.n += 1
		self.k_n[a] += 1

		# Update total
		self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
		
		
		# Update results for a_k
		if self.alpha == 1:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
		else:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) * self.alpha

		# if a >0:
		print('Action {0} reward {1}, All Reward {2}'.format(a, reward, self.k_reward))
		return a, reward, sm, ar_tab
	   
class ucb1_bandit(bandit):
	'''
	Upper Confidence Bound k-bandit problem
	 
	Inputs 
	============================================
	k: number of arms (int)
	c:
	iters: number of steps (int)
	mu: set the average rewards for each of the k-arms.
		Set to "random" for the rewards to be selected from
		a normal distribution with mean = 0. 
		Set to "sequence" for the means to be ordered from 
		0 to k-1.
		Pass a list or array of length = k for user-defined
		values.
	'''
	def __init__(self, k, c, iters, alpha = 1, reward = 0, fname = ''):
		bandit.__init__(self, k, iters, alpha, reward, fname)
		# Exploration parameter
		self.c = c
		self.k_ucb_reward = np.zeros(self.k)
		 
	def pull(self, a1, count = 0):
		# Select action according to UCB Criteria
		a = np.argmax(self.k_ucb_reward)
		return a

	def update_policy(self, a, t):
		# Execute the action and calculate the reward
		reward, sm, ar_tab = self.get_reward(a, t) #np.random.normal(self.mu[a], 1)
		
		# Update counts
		self.n += 1
		self.k_n[a] += 1

		# Update total
		self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
		 
		# Update results for a_k
		if self.alpha == 1:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
		else:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) * self.alpha
		
		self.k_ucb_reward = self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n)

		print('Action {0} reward {1}, All Reward {2}'.format(a, reward, self.k_ucb_reward))

		return a, reward, sm, ar_tab
	
def main(argv):
	
	inputfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:", ["ifile="])
	except getopt.GetoptError:
			print("MAB_BMC.py  -i ifile")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("MAB_BMC.py -i ifile")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
	print("Input file is :" , inputfile)

	k = 7 # arms
	iters = 1000 #int((TIMEOUT/T)) 
	#iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
	episodes = 1 #episodes
	print('iters', iters)
	alpha = 0.05
	reward = 0
	# Initialize bandits
	eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile)
	eps_01 = eps_bandit(k, 0.4, iters, 1, reward,  inputfile)
	eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile)
	eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile)
	eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile)
	ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile)

	eps_decay_alpha = eps_decay_bandit(k, iters, alpha, reward, inputfile)
	eps_high_alpha = eps_bandit(k, 0.0, iters, alpha, reward, inputfile)

	reward = 6000
	# o_eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile)
	o_eps_01 = eps_bandit(k, 0.4, iters, 1, reward,  inputfile)
	# o_eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile)
	# o_eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile)
	# o_eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile)
	# o_ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile)


	# options = [eps_0, eps_01, eps_1, eps_1_alpha, eps_decay, ucb1, o_eps_0, o_eps_01, o_eps_1, o_eps_1_alpha, o_eps_decay, o_ucb1]
	# labels = [r'$\epsilon=0$', r'$\epsilon=0.4$', r'$\epsilon=0.1$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha), r'$\epsilon-decay$', 'ucb1', \
	# r'opt $\epsilon=0$', r'opt $\epsilon=0.01$', r'opt $\epsilon=0.1$', r'opt $\epsilon=0.1, \alpha = {0}$'.format(alpha), \
	#   r'opt $\epsilon-decay$', r'opt ucb1']

	# options = [ eps_01, eps_1_alpha, eps_decay, ucb1, o_eps_01]
	# labels = [ r'$\epsilon=0.4$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha), r'$\epsilon-decay$', 'ucb1', r'opt $\epsilon=0.01$' ]

	options = [eps_high_alpha, ucb1]
	labels = [r'$\epsilon= 0.0, \alpha = {0}$'.format(alpha), 'ucb1']


	fname = (inputfile.split('/')[-1]).split('.')[0]
	print(fname)
	pp = PdfPages("plots/plot_MAB_BMC_D0_{0}_{1}{2}.pdf".format(fname, DIFF, '_FIX' if DIFF else ''))
	j = 0
	all_rewards = []
	all_selection = []
	all_results = []
	all_times = []
	for opt in options:

		# starting the monitoring
		tracemalloc.start()
		stime = time.time()
		print('---------------- Running bandit {0} ------------------'.format(labels[j]))
		rewards = np.zeros(iters)
		selection = np.zeros(k)
	
		# Run experiments
		for i in range(episodes):
			
			print('---- episodes ', i)
			# Run experiments
			res = opt.run()

			# Update long-term averages
			rewards = rewards + (opt.reward - rewards) / (i + 1)

			# Average actions per episode
			selection = selection + (opt.k_n - selection) / (i + 1)
		
		all_results.append(res)
		all_rewards.append(rewards)
		all_selection.append(selection)

		fig1 = plt.figure(figsize=(12,8))
		plt.plot(rewards, label=labels[j])
		plt.legend(bbox_to_anchor=(1.3, 0.5))
		plt.xlabel("Iterations")
		plt.ylabel("Average Reward")
		plt.title("Average Rewards after " + str(episodes)  + " Episodes")
		plt.legend()

		j += 1
		# plt.show()

		pp.savefig(fig1)   
		etime = time.time()
		# displaying the memory
		current, peak = tracemalloc.get_traced_memory()
		# stopping the library
		tracemalloc.stop()
		print('time by bandit ', labels[j-1], (etime-stime))
		print('Memory by bandit ', labels[j-1], (current/(1024*1024), peak/(1024*1024)))
		 
		all_times.append((etime-stime, current/(1024*1024), peak/(1024*1024)))
	print('-------------------------------------------')
	print()
	print('Bandit policy: \t BMC depth \t time \t sequence')
	
	all_plots = []
	fig2 = plt.figure()
	for j in range(len(options)):
		d, a, t, seq = all_results[j]
		print('{0}: \t {1} ({4}) \t time: {2} s, real: {5}s, Memory: {6}MB,{7}MB \t {3}'.format(labels[j], a if a > 0 else d, t, seq, 'assert' if a>0 else '', all_times[j][0], all_times[j][1], all_times[j][2]))
		plt.plot(all_rewards[j], label=labels[j])
		plt.legend(bbox_to_anchor=(1.3, 0.5))

		to_plot = [[],[]]
		for ss in seq:
			ac, tt, rw, tt, t, frame =  ss
			to_plot[0].append(frame)
			to_plot[1].append(rw)
		all_plots.append(to_plot)

	plt.xlabel("Iterations")
	plt.ylabel("Average Reward")
	plt.title("Average Rewards after " + str(episodes)  + " Episodes")
	plt.legend()
	pp.savefig(fig2)

	fig3 = plt.figure()
	for j in range(len(options)):
		to_plot = all_plots[j]
		plt.plot(to_plot[0], to_plot[1], label=labels[j])

	plt.xlabel("Frames")
	plt.ylabel("Reward")
	# plt.title("Average Rewards after " + str(episodes)  + " Episodes")
	plt.legend()
	pp.savefig(fig3)

	opt_per = np.array(all_selection)/ iters * 100
	df = pd.DataFrame(opt_per, index=labels, columns=[Actions[x] for x in range(0, k)])
	print("Percentage of actions selected:")
	print(df)
	
	pp.close()


	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
