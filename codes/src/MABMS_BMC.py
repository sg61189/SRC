import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
from scipy import interpolate
# from memory_profiler import memory_usage
from abcBMCUtil import *

DEBUG = True
DEBUG = False
OPT = True
T = 30 
TIMEOUT = 3600
SC = 2
# DIFF = 1 # BMC depth absolute
# DIFF = 0 # BMC depth relative
DIFF = 2 # function of depth, time, memory
# DIFF = 3 # function number of clauses/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
FIXED = False
PDR = False
MAX_FRAME = 1e4
MAX_CLAUSE = 1e9
MAX_TIME = 3600
TO = np.arange(30, 3600, 10)

Actions = ['bmc2', 'bmc3', 'bmc3rs', 'bmc3j', 'bmc3rg', 'bmcru', 'pdr']

class bandit:

    def __init__(self, k, iters, alpha, rewards=[], timeout = T, fname = ''):
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

        if len(rewards) > 0:
            for i in range(self.k):
                self.k_reward[i] = rewards[i]

        # recency constant
        self.alpha = alpha
        ##
        self.fname =  fname
        self.timeout = timeout #np.zeros(iters)


        # current state of the run; eg - bmc depth
        self.states = 0

    def get_reward(self, a, t1=-1):

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

        if t1 == -1:
            t = int(self.timeout)
        else:
            t = int(t1)

        if self.n == 0:
            simplify(fname, ofname)
            print('Simplified model', ofname)

        if a == 0:    #ABC bmc2
            asrt, sm, ar_tab = bmc2(ofname, sd, t)
        elif a == 1: #ABC bmc3
            asrt, sm, ar_tab = bmc3(ofname, sd, t)
        elif a == 2: #ABC bmc3rs
            asrt, sm, ar_tab = bmc3rs(ofname, sd, t)
        elif a == 3: #ABC bmc3x
            asrt, sm, ar_tab = bmc3j(ofname, sd, t)
        elif a == 4: #ABC bmc3az
            asrt, sm, ar_tab = bmc3rg(ofname, sd, t)
        elif a == 5: #ABC bmc3j
            asrt, sm, ar_tab = bmc3ru(ofname, sd, t)
        elif a == 6: #ABC pdr
            asrt, sm, ar_tab = pdr(ofname, t)

        if sm is not None:
            # print(sm)
            reward = 0
            if DIFF == 0 :
                if asrt > 0:
                    reward = 1/sm.tt
                else : #lif t - sm.tt > 0:
                    diff = (t - sm.tt) if (t - sm.tt) > 1.0 else 1.0
                    reward = (sm.frame - sd)/(1 + 0.1*sm.tt)

            elif DIFF == 1:
                reward = sm.frame
            elif DIFF == 2:
                #+ 0.2*sm.cla/MAX_CLAUSE 
                reward =  np.exp(0.3*sm.frame/MAX_FRAME - 0.2*sm.to/MAX_TIME)
                if asrt > 0:
                    #+ 0.2*sm.cla/MAX_CLAUSE 
                    reward =  np.exp(1 + 0.3*sm.frame/MAX_FRAME - 0.2*sm.to/MAX_TIME)
            else:
                reward = sm.cla/(10000 * sm.to) if sm.to > 0 else sm.to
        else:
            sm =  abc_result(frame=sd, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = t)
            if asrt > 0:
                reward = asrt
            else:
                reward = 0
        # print('reward', reward, 'sm', sm)

        return reward, sm, ar_tab


    def run(self, stage = 0):
        seq = []
        sm = None
        next_to = -1

        frames = []
        time_outs = []
        for i in range(self.iters):
          
            a, reward, sm, ar_tab = self.pull()
        
            # if sm:
            #     ss = (Actions[a], self.timeout, reward, totalTime, self.states)
            # #   # seq.append()
            # else:
            #    ss = (a, -1, reward, -1, self.states)
            seq.append((a, sm.tt, sm.frame, sm.asrt))
            self.reward[i] = self.mean_reward

            #if sm and sm.to > 0:
            print('iter ', i, Actions[a], 'reward', reward, self.timeout, sm)

            # self.timeout[i] = T
            

        # while i < self.iters:
        #     self.reward[i] = self.mean_reward
        #     i += 1
        # print('BMC-depth reached at stage', stage, sm.frame, 'totalTime', totalTime)
        # print('Sequence of actions and BMC depth:', seq)
        return seq
            
    def reset(self, reward):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        for i in range(self.k):
            self.k_reward[i] = reward
        self.timeout = 0

    
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
    def __init__(self, k, c, iters, alpha = 1, rewards = [], timeout = T, fname = ''):
        bandit.__init__(self, k, iters, alpha, rewards, timeout, fname)
        # Exploration parameter
        self.c = c
        self.k_ucb_reward = np.zeros(self.k)
         
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_ucb_reward)

        # time_out = np.random.choice(TO, size=1)[0]
        time_out = np.random.normal(self.timeout, self.timeout * 0.1, 1)[0]

        # Execute the action and calculate the reward
        reward, sm, ar_tab = self.get_reward(a, time_out) #np.random.normal(self.mu[a], 1)
        
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

        print('ucb1-pull', 'k_reward', self.k_reward, 'extra', self.c * np.sqrt((np.log(self.n)) / self.k_n), 'k_ucb_reward', self.k_ucb_reward)

        return a, reward, sm, ar_tab
    
def main(argv):
    
    inputfile = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:", ["ifile="])
    except getopt.GetoptError:
            print("MABMS_BMC.py  -i ifile")
            sys.exit(2)
            
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("MABMS_BMC.py -i ifile")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print("Input file is :" , inputfile)

    k = 6 # arms
    iters = 50 ##int((TIMEOUT/T)) 
    #iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
    episodes = 1 #episodes
    print('iters', iters)
    reward = 1000
    c = 2

    labels = [r'opt ucb1']

    fname = (inputfile.split('/')[-1]).split('.')[0]
    print(fname)
    pp = PdfPages("plots/plot_MABMS_{0}_{1}.pdf".format(DIFF, fname))
    
    stage = 0
    totalTime = 0
    states = 0
    depth = 0
    seq = []
    all_results = []
    all_select = []
    timeout = T
    sequence = []

    s_rewards = []
    while True:
        ucb1_rewards = np.zeros(iters)
        ucb1_selection = np.zeros(k)
        max_reward = 0

        print('---- stages --- ', stage, inputfile, 'timeout', timeout, 'totalTime', totalTime)
        # Run experiments
        #for i in range(episodes):
            
        # Initialize bandits
        ucb1 = ucb1_bandit(k, c, iters, 1,  s_rewards, timeout, inputfile)
        ucb1.states = states
        
        # Run experiments
        res = ucb1.run(stage)
        # sm.frame, sm.asrt, totalTime
        sel = res

        # Update long-term averages
        ucb1_rewards = ucb1_rewards + (ucb1.reward - ucb1_rewards) #/ (i + 1)

        # Average actions per episode
        ucb1_selection = ucb1_selection + (ucb1.k_n - ucb1_selection) #/ (i + 1)

        max_action = np.argmax(ucb1_selection)

        min_t = timeout
        depth = 0
        asrt = -1
        for ss in sel:
            if ss[0] == max_action:
                if min_t > ss[1]:
                    min_t = ss[1]
                    depth = ss[2]
                    asrt = ss[3]

        sequence.append((max_action, depth, min_t))

        all_results.append(res)
        all_select.append(max_action)

        print('##### @@@@ Stage {0} action {1} time {2}'.format(stage, max_action, ucb1.timeout))
        states = depth+1 if depth > 0 else depth

        totalTime += ucb1.timeout #self.timeout[i]

        timeout = min(max(ucb1.timeout * SC, 480), TIMEOUT - totalTime)
        # reward = max_reward
        stage = stage + 1
        iters = max(1, math.ceil(iters * 0.75))
        s_rewards = ucb1_rewards

        fig1 = plt.figure(figsize=(12,8))
        plt.plot(ucb1_rewards, label=labels[0])
        plt.legend(bbox_to_anchor=(1.3, 0.5))
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward at stage {0}".format(stage))
        plt.title("Average Rewards after " + str(episodes)  + " Episodes")
        plt.legend()

        pp.savefig(fig1)   


        if totalTime >= TIMEOUT:
            print('---------------------END TIMEOUT RESULT------------------ ')
            print('BMC-depth reached ', depth, 'totalTime', totalTime)
            print('Stopping iteration')
            break

        if asrt > 0:
            print('---------------------END ASSERTION RESULT------------------ ')
            print('Output asserted at frame ', asrt, 'totalTime', totalTime)
            print('Stopping iteration')
            break
        
    print('-------------- SEQUENCE -----------------------------')
    print(','.join(['[{0}, {1}, {2}]'.format(Actions[a[0]], a[1], a[2]) for a in sequence]))
    # print('Bandit policy: \t BMC depth \t time \t sequence')
    # fig2 = plt.figure(figsize=(12,8))
    # for j in range(len(stages)):
    #     d, a = all_results[j]
    #     print('{0}: \t {1} ({4}) \t {2} s \t {3}'.format(labels[j], a if a > 0 else d, t, s, 'assert' if a>0 else ''))
    #     plt.plot(all_rewards[j], label=labels[j])
    #     plt.legend(bbox_to_anchor=(1.3, 0.5))
    # plt.xlabel("Iterations")
    # plt.ylabel("Average Reward")
    # plt.title("Average Rewards after " + str(episodes)  + " Episodes")
    # plt.legend()
    # pp.savefig(fig2)

    # opt_per = np.array(all_selection)/ iters * 100
    # df = pd.DataFrame(opt_per, index=labels, columns=[Actions[x] for x in range(0, k)])
    # print("Percentage of actions selected:")
    # print(df)
    
    pp.close()


    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
