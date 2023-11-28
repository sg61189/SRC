import numpy as np 
PLOT = False
if PLOT:
	import matplotlib.pyplot as plt 
	from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math, csv
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
TIMEOUT = 3600#/2.0
SC = 2
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
# DIFF = 2 # function of depth, time, memory
DIFF = 3 # function number of frames/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
FIXED = False
FIXED = True
PDR = False
MAX_FRAME = 1e9
MAX_CLAUSE = 1e9
MAX_TIME = 3600
MAX_MEM = 2000
MAX_TIMEOUT = 4*MAX_TIME
F = 3

time_outs = {}
Actions = ['bmc2', 'bmc3', 'bmc3rs', 'bmc3j', 'bmc3rg', 'bmcru', 'bmc3r', 'pdr']
To1 = []
st = 60
tot = 0
Iters = 0
while tot < TIMEOUT:
	To1.append(st)
	tot += st
	st = min(st*SC,  TIMEOUT - tot)
	Iters += 1

M = int(len(Actions))

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
		#for i in range(iters):
		self.timeout[0] = To1[0]
		
		self.engine_res = [{} for i in range(k)]


		# current state of the run; eg - bmc depth
		self.states = 0

	def get_reward(self, a, t1 = -1):

		# get starting depth
		# sd = 0
		# if self.n  == 0:
		sd = int(self.states)+1 if int(self.states) > 0 else int(self.states) 

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
			asrt, sm, ar_tab, tt1 = bmc2(ofname, sd, t=t)
		elif a == 1: #ABC bmc3
			asrt, sm, ar_tab, tt1 = bmc3(ofname, sd, t=t)
		elif a == 2: #ABC bmc3rs
			asrt, sm, ar_tab, tt1 = bmc3rs(ofname, sd, t=t)
		elif a == 3: #ABC bmc3j
			asrt, sm, ar_tab, tt1 = bmc3j(ofname, sd, t=t)
		elif a == 4: #ABC bmc3g
			asrt, sm, ar_tab, tt1 = bmc3rg(ofname, sd, t=t)
		elif a == 5: #ABC bmc3u
			asrt, sm, ar_tab, tt1 = bmc3ru(ofname, sd, t=t)
		elif a == 6: #ABC bmc3r
			asrt, sm, ar_tab, tt1 = bmc3r(ofname, sd, t=t)
		elif a == 7: #ABC pdr
			asrt, sm, ar_tab, tt1 = pdr(ofname, t)
		
		# min_t = MAX_TIME
		# if len(ar_tab.keys()) > 0:
        # 	min_k = sorted(ar_tab.keys())[0]
        # 	min_t =  ar_tab[min_k].to

		ar_tab_old = self.engine_res[a]
		for ky in ar_tab.keys():
			sm1 = ar_tab[ky]
			if sm1 and  sm1.frame > sd:
				sm = sm1
				ar_tab_old.update({ky:sm})

		self.engine_res[a] = ar_tab_old
		MAX_mem = 0
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
				c1 = 1.0/MAX_MEM
				c2 = 1.0/MAX_TIME
				pen = t - sm.tt
				reward = 0
				cn = 0
				for ky in ar_tab_old.keys():
					tm = ar_tab_old[ky].to
					mem = ar_tab_old[ky].mem
					#reward += 2*np.exp(-c1*mem/(1+ky) - c2*tm/(1+ky))    
					reward += (1*tm/(1+ky))
					cn += 1
				wa = reward/cn
				reward = 2*np.exp(-wa)#(reward + np.exp(-pen/MAX_TIME))/cn
				if sd > sm.frame:
					reward = -1 * np.exp(t/MAX_TIME)
		else:
			sm =  abc_result(frame=sd, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = tt1,ld= sd)
			if asrt > 0:
				reward = asrt
			else:
				reward = -1 * np.exp(t/MAX_TIME) #np.log(t)
		print(sd, sm.frame, reward, sm)
		#todo: clean up the directory and store it in a file
		# for k in list(ar_tab_old):
		#     v = ar_tab_old[k]
		#     if not v:
		#         del ar_tab_old[k]
		return reward, sm

	def run(self):
		totalTime = 0
		all_time = 0
		seq = []
		sm = None
		next_to = -1
		next_time = {}
		# for i in self.iters:
		#     next_time[i] = -1
		for a in range(self.k):
			time_outs.update({a:next_time})

		# def Next(d, a):
		# 	sd = d+1 if d > 0 else d  
		# 	# if a == 0:
		# 	# 	sd = d
		# 	return sd

		frames = {}
		count = 0
		best_sd = 0
		best = ()
		end_frame = ()
		asrt = -1

		repeat_count = 2*self.k

		enter_critical = False
		exit_critical = True
		critical = False

		ocount = 0
		ending = 0
		conf_begin_phase = 0
		max_conf = 0
		explore_count = 0
		all_ending = 0

		a = 0
		ecount = 0
		MAX_mem = 0
		for i in range(4*self.iters):

			if all_ending or int(1.25*MAX_TIMEOUT - all_time) <= 0:		
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('BMC-depth reached ', self.states, 'totalTime', totalTime, 'all_time', all_time)
				print('Stopping iteration -- all timeout')
				break
				
			if int(TIMEOUT - totalTime) <= 0:
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('BMC-depth reached ', self.states, 'totalTime', totalTime)
				print('Stopping iteration -- seq timeout')
				break

			print('Iteration {0} start ------'.format(i))

			if i < (repeat_count):
				a = i%(self.k)
				print('exploring select', a)
				if i > 0:
					self.timeout[i] = self.timeout[i-1]
				# if i%self.k == self.k-1:
				# 	self.timeout[i] = self.timeout[i-1] * SC
				# ocount += 1
			else:
				if i > 1 and enter_critical:	
					a = i%(self.k)
					print('critical exploring select', a)
					self.timeout[i] = self.timeout[i-1]
					ocount += 1
				else:
					a = self.pull(a, count)
					next_timeout = self.timeout[i-1] * SC
					# self.timeout[i] = (self.timeout[i-1] * SC)#,  TIMEOUT - totalTime)#, 480)
					# if sm and sm.to == -1 and count > int(M/2):
					self.timeout[i] = next_timeout
					# else:

					if ending : #or self.timeout[i-1] > next_timeout > TIMEOUT - (totalTime + self.timeout[i]):
						if ecount > self.k-1:
							end_frame = self.states, asrt, totalTime, seq, MAX_mem
							print('BMC-depth reached ', self.states, 'totalTime', totalTime)
							print('Stopping iteration - condition with timeout ', next_timeout)
							break
						else:
							ecount += 1

				print('Calculating time out', self.timeout[i], 'total till now', totalTime)

				if self.timeout[i] > TIMEOUT - (totalTime + self.timeout[i]):
					self.timeout[i] =  (TIMEOUT - totalTime) #self.timeout[i] +
					ending = 1
				else:
					self.timeout[i] = min(self.timeout[i], TIMEOUT - totalTime)
			
			if int(MAX_TIMEOUT - all_time) <= 0:		
				a = self.pull(a, count=2)
				self.timeout[i] = min(0.25*MAX_TIMEOUT, TIMEOUT - totalTime)
				print('More than {0} hrs spent in learning --- closing iterations now'.format(MAX_TIMEOUT/TIMEOUT))
				all_ending = True
				enter_critical = False
				exit_critical = True
				ocount = 0
				#break
				# if self.timeout[i] > 3000:
				# 	self.stt

			print('Next time out', self.timeout[i], 'for chosen action', a, Actions[a], 'ocount', ocount, 'enter_critical', enter_critical, 'exit_critical', exit_critical, 'critical', critical, 'ending', ending)

			if self.timeout[i] < 1.0:
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('BMC-depth reached ', self.states, 'totalTime', totalTime)
				print('Stopping iteration - condition with timeout < ', 1.0)
				break

			a, reward, sm = self.update_policy(a, self.timeout[i])

			if sm :
				if MAX_mem < sm.mem :
					MAX_mem = sm.mem
			# fragmentation
			tt = math.ceil(sm.tt) #sm.tt if sm.asrt > 0 else self.timeout[i] #sm.tt if sm.asrt > 0  else math.ceil(sm.tt) # self.timeout[i]

			all_time += sm.tt if sm.asrt > 0 else self.timeout[i]

			if not all_ending and sm and sm.to == -1:# and self.timeout[i] >= 120:
				if not enter_critical and i > repeat_count:
					critical = True # blocker 
					print('blocker -- critical phase')
				else:
					if i < repeat_count:
						count += 1
						if count > int(M/2):
							critical = True # blocker 
							print('blocker -- critical phase')
			else:
				count = 0

			sd = self.states
			if sm and reward > 0:
				count = 0
				print(i, 'sm', 'conf', sm.conf, 'cla', sm.cla, max(F*conf_begin_phase, 1e5), 'conf_begin_phase', conf_begin_phase, 'ocount', ocount, 'enter_critical', enter_critical, 'exit_critical', exit_critical, 'critical', critical, 'iter', (i+1)%self.k, 'repeat_count', repeat_count, 'M', M)

				sd = sm.ld #Next(sm.ld, a)
				ss = (Actions[a], tt, reward, totalTime, self.timeout[i], sd)

				if len(best) == 0:
					best = ss

				if (not all_ending) and i > repeat_count and (sm.cla > max(F*conf_begin_phase, 1e5)):
					if not enter_critical:
						critical = True # clauses incresed
						print('clauses incresed -- critical phase')

				if (i < repeat_count ) or (enter_critical)  : # exploration
					sd = sm.ld #Next(sm.ld, a)
					if best_sd < sd:
						best_sd = sd
						best = ss

					elif best_sd == sd:
						if best[1] > sm.tt:
							best_sd = sd
							best = ss
					#best_sd = max(best_sd, sd)
					# if (i < repeat_count):
					max_conf = max(max_conf, sm.cla)

					print('------ exploring --', i, 'critical', critical, 'best_sd', best_sd, 'max_conf', max_conf)
					if (i < repeat_count and i%self.k == self.k-1) or (enter_critical and ocount >= M-1) or (i < repeat_count and sm.asrt > 0) or (enter_critical and sm.asrt > 0):
						# end of exploration --- pick the best one
						print('------ at the end of exploration')
						# sd = sm.frame+1 if sm.frame > 0 else sm.frame
						# best_sd = max(best_sd, sd)
						if self.states < best_sd:		
							self.states = best_sd
							ss = best #(Actions[a], tt, reward, totalTime, self.timeout[i], self.states)
							seq.append(ss)
							totalTime += tt
						# if (i < repeat_count and i%self.k == self.k-1):
						conf_begin_phase = max_conf

				elif all_ending or (exit_critical and not enter_critical and i >= repeat_count): # exploitation 				
					print('------ no exploration')
					sd = sm.ld #Next(sm.ld, a) 
					self.states = sd
					ss = (Actions[a], tt, reward, totalTime, self.timeout[i], sd)
					seq.append(ss)
					totalTime += tt

			else:
				ss = (Actions[a], -1, reward, -1, self.timeout[i], sd)
				if all_ending:
					totalTime += tt


			if critical and not enter_critical and i > repeat_count:
				enter_critical = True
				exit_critical = False
				ocount = 0
				explore_count += 1
				print('----- Started exploring --', i, ocount, 'explore_count', explore_count)

			if enter_critical and (ocount >= M-1): #(i+1)%self.k == self.k-1:
				enter_critical = False
				exit_critical = True
				critical = False
				print('----- Stopped exploring --', i, ocount)

			
			self.reward[i] = self.mean_reward

			# if sm and sm.to > 0 and reward > 0:
			#     totalTime += tt
				
			print('#### iter ', i, Actions[a], 'time taken', tt, self.timeout[i], 'totalTime', totalTime, 'ss', ss, sm)

			print('Iteration {0} end ------'.format(i))

			# self.timeout[i] = T
			if sm and sm.asrt > 0:
				asrt = sm.asrt
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('Output asserted at frame ', sm.asrt, 'tt', tt, 'totalTime', totalTime)
				print('Stopping iteration')
				break

		# while i < self.iters:
		# 	self.reward[i] = self.mean_reward
		# 	i += 1
		print(i, 'BMC-depth reached ', self.states, 'totalTime', totalTime)
		print(i, 'Sequence of actions and BMC depth:', seq)
		return end_frame #(sm.frame, sm.asrt, totalTime, seq)
			
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
			print('selection - cond1',a, p, self.n, self.eps, a1, count)
		if count > 0:
			ap = []
			for i in range(self.k):
				if not(i == a1):
					ap.append(i)
			a = np.random.choice(ap)
			print('selection - cond2',a, p, self.n, self.eps, a1, count)
		elif p < self.eps:
			# Randomly select an action
			a = np.random.choice(self.k)
			print('selection - cond3',a, p, self.n, self.eps, a1, count)
		else:
			# Take greedy action
			a = np.argmax(self.k_reward)
			print('selection - cond4',a, p, self.n, self.eps, a1, count)
		return a

	def update_policy(self, a, t):
		# Execute the action and calculate the reward
		# mem_use, tm = memory_usage(self.get_reward(a))
		reward, sm = self.get_reward(a, t)
		
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
		return a, reward, sm
		
   
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
		elif count > 0:
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
		reward, sm = self.get_reward(a, t)
		
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
		return a, reward, sm
	   
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
		reward, sm = self.get_reward(a, t) #np.random.normal(self.mu[a], 1)
		
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

		return a, reward, sm
	
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

	fname = (inputfile.split('/')[-1]).split('.')[0]
	print(fname)
	filename = "plots_IF/MAB_BMC_results_fixt_IF_{0}_{1}.csv".format(TIMEOUT, fname)
	# # header = ['Design', 'Frame', 'Clauses', 'Mem', 'time']
	# # writing to csv file 
	# with open(filename, 'w+') as csvfile: 
	# 	print('filename', inputfile)

	k = 7 # arms
	iters = 1000 #int((TIMEOUT/T)) 
	#iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
	episodes = 1 #episodes
	print('iters', iters)
	alpha = 0.6
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

	options = [eps_high_alpha]
	labels = [r'$erwa$'.format(alpha)]

	if PLOT:
		pp = PdfPages("plots_IF/plot_MAB_BMC_fixn_IF_{0}_{1}{2}.pdf".format(fname, DIFF, '_FIX' if DIFF else ''))
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

		if PLOT:
			fig1 = plt.figure(figsize=(12,8))
			plt.plot(rewards, label=labels[j])
			plt.legend(bbox_to_anchor=(1.3, 0.5))
			plt.xlabel("Iterations")
			plt.ylabel("Average Reward")
			plt.title("Average Rewards after " + str(episodes)  + " Episodes")
			plt.legend()
			pp.savefig(fig1)   

		j += 1
		# plt.show() 
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

	rows  = []
	
	rows.append(['Design', 'Bandit-policy','BMC-depth', 'status', 'time(s)', 'total(s)', 'memory-current(MB)', 'memory-peak MB)', 'MAX-mem(BMC)(MB)', 'sequence'])
	print('Bandit policy: \t BMC depth \t time \t sequence')
	
	all_plots = []
	if PLOT:
		fig2 = plt.figure()
	for j in range(len(options)):
		d, a, t, seq, mem = all_results[j]
		print('{0}: \t {1} ({4}) \t time: {2:0.2f} s, real: {5:0.2f}s, Memory: {6:0.2f}MB,{7:0.2f}MB {8}MB \t {3} '.format(labels[j], a if a > 0 else d, t, seq, 'assert' if a>0 else '', all_times[j][0], all_times[j][1], all_times[j][2], mem))
		rows.append([fname, labels[j], a if a > 0 else d, 'sat' if a>0 else 'timeout', '{0:0.2f}'.format(t), '{0:0.2f}'.format(all_times[j][0]), '{0:0.2f}'.format(all_times[j][1]), mem, '{0:0.2f}'.format(all_times[j][2]), seq])

		if PLOT:
			plt.plot(all_rewards[j], label=labels[j])
			plt.legend(bbox_to_anchor=(1.3, 0.5))

		to_plot = [[],[]]
		for ss in seq:
			ac, tt, rw, tt, t, frame =  ss
			to_plot[0].append(frame)
			to_plot[1].append(rw)
		all_plots.append(to_plot)

	if PLOT:
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

		pp.close()

	opt_per = np.array(all_selection)/ iters * 100
	df = pd.DataFrame(opt_per, index=labels, columns=[Actions[x] for x in range(0, k)])
	print("Percentage of actions selected:")
	print(df)
	

	with open(filename, 'a+') as csvfile: 
		csvwriter = csv.writer(csvfile) 
		csvwriter.writerows(rows)

	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
