import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
# from memory_profiler import memory_usage
from abcBMCUtil import *

DEBUG = True
DEBUG = False
OPT = True
T = 30 
TIMEOUT = 3600
SC = 2
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
DIFF = 2 # function of depth, time, memory
# DIFF = 3 # function number of clauses/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
PDR = False
MAX_FRAME = 1e4
MAX_CLAUSE = 1e10
MAX_TIME = 3600



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


        # current state of the run; eg - bmc depth
        self.states = 0

    def get_reward(self, a):

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

        t = int(self.timeout[self.n])

        if self.n == 0:
            simplify(fname, ofname)
            print('Simplified model', ofname)

        if a == 0:    #ABC bmc2
            asrt, sm = bmc2(ofname, sd, t)
        elif a == 1: #ABC bmc3
            asrt, sm = bmc3(ofname, sd, t)
        elif a == 2: #ABC bmc3s
            asrt, sm = bmc3s(ofname, sd, t)
        elif a == 3: #ABC bmc3j
            asrt, sm = bmc3j(ofname, sd, t)
        elif a == 4: #ABC bmc3az
            asrt, sm = bmc3az(ofname, sd, t)
        elif a == 5: #ABC bmc3x
            asrt, sm = bmc3x(ofname, sd, t)
        elif a == 6: #ABC pdr
            asrt, sm = pdr(ofname, t)

        if sm is not None:
            # print(sm)
            reward = 0
            if DIFF == 0 :
                reward = (sm.frame - sd)
            elif DIFF == 1:
                reward = sm.frame
            elif DIFF == 2:
                reward =  np.exp(0.3*sm.frame/MAX_FRAME + 0.2*sm.cla/MAX_CLAUSE - 0.5*sm.to/MAX_TIME)
            else:
                reward = sm.cla/(10000 * sm.to) if sm.to > 0 else sm.to
            self.states = sm.frame+1 if sm.frame > 0 else sm.frame
        else:
            sm =  abc_result(frame=sd-1, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt)
            if asrt > 0:
                reward = asrt
            else:
                reward = 0
        print(reward, sm)

        return reward, sm


    def run(self):
        totalTime = 0
        seq = []
        sm = None
        for i in range(self.iters):
            if totalTime > TIMEOUT:
                print('BMC-depth reached ', self.states)
                print('Stopping iteration')
                break
            if i == 0:
                self.timeout[i] = T
            else:
                if sm and sm.to == -1:
                    self.timeout[i] = self.timeout[i-1] * SC 
                else:
                    self.timeout[i] = self.timeout[i-1]

            a, reward, sm = self.pull()
        
            if sm:
                ss = (a, reward, self.states, sm.to)
               # seq.append()
            else:
                ss = (a, reward, self.states, -1)
            seq.append(ss)
            print('iter ', i, ss, self.timeout[i], totalTime, sm)
            self.reward[i] = self.mean_reward

            if sm and sm.to > 0:
                totalTime += self.timeout[i]

            # self.timeout[i] = T
            if sm and sm.asrt > 0:
                print('Output asserted at frame ', sm.asrt)
                print('Stopping iteration')
                break
        print('Sequence of actions and BMC depth:', seq)
            
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
        
    def pull(self):
        # Generate random number
        # np.random.seed(time.time())
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
        
        # Execute the action and calculate the reward

        # mem_use, tm = memory_usage(self.get_reward(a))
        reward,sm = self.get_reward(a)
        
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
        #print('For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
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
       
    def pull(self):
        # Generate random number
        # np.random.seed(self.n)
        p = np.random.rand()
        if p < 1 / (1 + self.n / self.k):
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
        
        # Execute the action and calculate the reward

        # mem_use, tm = memory_usage(self.get_reward(a))
        res, sm = self.get_reward(a)

        reward = res #max(0, 1/tm) ##-1*mem_use 
        # if DEBUG:
        # reward = np.random.normal(self.mu[a], 1)
        
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
        # print('For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
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
         
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_ucb_reward)
             
        reward,sm = self.get_reward(a) #np.random.normal(self.mu[a], 1)
        
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
        
        self.k_ucb_reward[a] = self.k_reward[a] + self.c * np.sqrt((np.log(self.n)) / self.k_n[a])
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

    k = 5 # arms
    iters = int((TIMEOUT/T)) 
    #iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
    episodes = 1 #episodes
    print('iters', iters)
    alpha = 0.05
    reward = 0
    # Initialize bandits
    eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile)
    eps_01 = eps_bandit(k, 0.01, iters, 1, reward,  inputfile)
    eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile)
    eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile)
    eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile)
    ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile)

    reward = 100
    o_eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile)
    o_eps_01 = eps_bandit(k, 0.01, iters, 1, reward,  inputfile)
    o_eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile)
    o_eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile)
    o_eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile)
    o_ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile)

    options = [eps_0, eps_01, eps_1, eps_1_alpha, eps_decay, ucb1, o_eps_0, o_eps_01, o_eps_1, o_eps_1_alpha, o_eps_decay, o_ucb1]
    labels = [r'$\epsilon=0$', r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha), r'$\epsilon-decay$', 'ucb1', \
    r'opt $\epsilon=0$', r'opt $\epsilon=0.01$', r'opt $\epsilon=0.1$', r'opt $\epsilon=0.1, \alpha = {0}$'.format(alpha), \
        r'opt $\epsilon-decay$', r'opt ucb1']

    options = [eps_0, eps_01, eps_1, eps_1_alpha, eps_decay, ucb1, o_eps_01]
    labels = [r'$\epsilon=0$', r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha), r'$\epsilon-decay$', 'ucb1', r'opt $\epsilon=0.01$' ]

    fname = (inputfile.split('/')[-1]).split('.')[0]
    print(fname)
    pp = PdfPages("plots/plot_MAB_BMC_all_{0}_{1}{2}.pdf".format(fname,T, '_DIFF' if DIFF else ''))
    j = 0
    all_rewards = []
    all_selection = []
    for opt in options:

        print('---------------- Running bandit {0} ------------------'.format(labels[j]))
        rewards = np.zeros(iters)
        selection = np.zeros(k)
    
        # Run experiments
        for i in range(episodes):
            
            print('---- episodes ', i)
            # Run experiments
            opt.run()
            
            # Update long-term averages
            rewards = rewards + (opt.reward - rewards) / (i + 1)


            # Average actions per episode
            selection = selection + (opt.k_n - selection) / (i + 1)
        
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

    fig2 = plt.figure(figsize=(12,8))
    for j in range(len(options)):
        plt.plot(all_rewards[j], label=labels[j])
        plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards after " + str(episodes)  + " Episodes")
    plt.legend()
    pp.savefig(fig2)

    opt_per = np.array(all_selection)
    df = pd.DataFrame(opt_per, index=labels, columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)
    
    pp.close()


    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
