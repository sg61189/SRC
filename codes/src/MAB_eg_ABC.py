import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from memory_profiler import memory_usage
from collections import namedtuple

PATH = "../"
DEBUG = True
DEBUG = False

# MAB policy to select
EPS_GREEDY = True
# EPS_GREEDY = False # for UCB1

# to determine timeout based on Luby sequence
LUBY = True
LUBY = False

T = 10

abc_result =  namedtuple('abc_result', ['frame', 'var', 'cla', 'conf', 'to']) 


class eps_bandit:
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
    
    def __init__(self, k, eps, iters, timeout = T, fname = '', mu='random'):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
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
        # storing the selected value at every run
        self.selections = np.zeros(k)

        # current state of the run; eg - bmc depth
        self.states = np.zeros(k)

        # mean reward in UCB1 policy
        self.k_reward_max = np.zeros(k)

        # current timeout value
        self.timeout = np.zeros(iters)

        ## file name 
        self.fname =  fname
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            # self.mu = np.random.normal(0, 1, k)
            self.mu = np.random.uniform(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
    
    def get_reward(self, a):
        start_time = time.time()

        # get starting depth
        # sd = 0
        # if self.n  == 0:
        sd = int(self.states[a])

        pname = os.path.join(PATH, "ABC")
        cmdName = os.path.join(pname, "abc")
        fname = os.path.join(PATH, self.fname)
        st = ''

        t = int(self.timeout[self.n])

        if a == 0:    #ABC bmc2
            command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; \
            &get; bmc2 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(fname, sd, t)
            st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]

        elif a == 1: #ABC bmc3
            command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; \
            bmc3 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(fname, sd, t)
            st = ' '.join([cmdName, "-c", command])  #, "--boound", "20"]

        elif a == 2: #ABC &bmc
             command = "\"read {0}; print_stats; &get; &bmc -S {1} -T 10 -v; print_stats\"".format(fname, sd)
             st = ' '.join([cmdName, "-c", command])  #, "--boound", "20"]

        elif a == 3: #abc pdr    
            command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; \
            pdr -v -T {1:5d}; print_stats\"".format(fname, t)
            st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
        
        elif a == 4:  #ABC reach
            command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; \
            reach -T {1:5d} -F 1 -v -L stdout; print_stats\"".format(fname, t)
            st = ' '.join([cmdName, "-c", command])  #, "--boound", "20"]
        
        if DEBUG:
            print(a,'\t----- '+str(st))

        # p = subprocess.run(st) #, capture_output=True)
        try:
            # gc.collect()
            output =  subprocess.run(st, shell=True, capture_output = True, text = True).stdout#.strip("\n")
            #subprocess.check_output(st)#, timeout=6*3600)
            out = 0
        except subprocess.CalledProcessError as e:
            out = e.returncode
            output = e.stdout
        except Exception as e:
            print('Running call again....') 
            if DEBUG:
                print('\t----- '+str(st))
            try:
                output =  subprocess.run(st, shell=True, capture_output = True, text = True).stdout#.strip("\n") #subprocess.check_output(st)
                out = 0
            except subprocess.CalledProcessError as e:
                out = e.returncode
                output = e.stdout
        if DEBUG:
            print('Cmd res:', out, output)
        
        end_time = time.time()

        '''This will give you the output of the command being executed'''
        if DEBUG:
            print ("\t----- Output: " + str(out),  'out', output)       
        
        sys.stdout.flush()
        
        sm = None
        res = -1, None
        if(out == 0):
            if DEBUG:
                print(output)
            sm2 = []

            # timeout
            to = t

            # if the engine is aborted, then timeout in -1
            if ( 'Aborted' in output):
                to = -1

            sm1 = [0,0,0,0,0]
           
            # parse the output  to collect information such as bmc depth reached (frame), timeout (to) and other circuit state

            if (a == 0): #bmc2
                xx = r'[ \t]+([\d]+)[ \t]+[:][ \t]+F[ \t]+=[ \t]+([\d]+).[ \t]+O[ \t]+=[ \t]+([\d]+).[ \t]+And[ \t]+=[ \t]+([\d]+).[ \t]+Var[ \t]+=[ \t]+([\d]+).[ \t]+Conf[ \t]+=[ \t]+([\d]+)[.]*'
                m = re.finditer(xx, output, re.M|re.I)
                for m1 in m:
                    sm1 = int(m1.group(1)), int(m1.group(5)), int(m1.group(4)), int(m1.group(6))
                    if DEBUG:
                        print(sm1)      
                    sm = abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], to=to) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                res =  (end_time - start_time), sm

            elif (a == 1): #bmc3
                xx = r'[ \t]*([\d]+)[ \t]+[+][ \t]+[:][ \t]+Var[ \t]+=[ \t]+([\d]+).[ \t]+Cla[ \t]+=[ \t]+([\d]+).[ \t]+Conf[ \t]+=[ \t]+([\d]+).[ \t]+Learn[ \t]+=[ \t]+([\d]+)[.]*'
                m = re.finditer(xx, output, re.M|re.I)
                for m1 in m:
                    sm1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3)), int(m1.group(4))
                    if DEBUG:
                        print(sm1)   
                    sm =  abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], to=to) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                res =  (end_time - start_time), sm
            elif (a == 2): # &bmc
                # print(output)
                xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
                m = re.finditer(xx, output, re.M|re.I)
                # print(m)
                for m1 in m:
                    sm1 = int(m1.group(1)), float(m1.group(2))
                    # print(sm1)   
                    sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, to=sm1[0])  #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                    #print(sm)
                sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, to=sm1[0])
                res =  (end_time - start_time), sm
            # elif (a == 3): # pdr
            #     xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
            #     m = re.finditer(xx, output, re.M|re.I)
            #     for m1 in m:
            #         sm1 = int(m1.group(1)), float(m1.group(2))
            #         if DEBUG:
            #             print(sm1)  
            #         sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, to=sm1[0]) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
            #     res =  (end_time - start_time), sm
            # elif (a == 4): # reach
            #     xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
            #     m = re.finditer(xx, output, re.M|re.I)
            #     for m1 in m:
            #         sm1 = int(m1.group(1)), float(m1.group(2))
            #         if DEBUG:
            #             print(sm1)
            #         sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, to=sm1[0]) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
            #     res =  (end_time - start_time), sm
        return res


    def pull(self):
        # Generate random number
        p = np.random.rand()
        if EPS_GREEDY:
            if self.eps == 0 and self.n == 0:
                a = np.random.choice(self.k)
            elif p < self.eps:
                # Randomly select an action
                a = np.random.choice(self.k)
            else:
                # Take greedy action with probability (1 - eps)
                a = np.argmax(self.k_reward)
        else: # for UCB1
            if self.eps == 0 and self.n == 0:
                a = np.random.choice(self.k)
            # elif p < self.eps:
            #     # Randomly select an action
            #     a = np.random.choice(self.k)
            else:
                # Take greedy action following UCB1 policy
                a = np.argmax(self.k_reward_max)
        
        # Execute the action and calculate the reward
        tm, sm = self.get_reward(a)

        if sm is None:  # default reward function as 1/(time duration) when no timeout is given
            print('sm None', sm)
            reward = 1/tm ##-1*mem_use 
            #or (self.timeout[self.n] == 0):
        else: # reward is calculated by number of frames checked by the selected engine
            print('sm = ', sm)
            reward = sm.frame #+ self.states[a] 
            # store the maximum frame number checked as current state
            self.states[a] = sm.frame - 1 if sm.frame > 0 else sm.frame
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        
        # Update results for a_k -- stationary
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

        # # # Update results for a_k -- non-stationary
        # self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.n

        # calculate reward function of UCB1 policy
        self.k_reward_max[a] = self.k_reward[a] + np.sqrt(8*np.log(self.n)/self.k_n[a])

        print('------------ ', self.n, self.k_n[a], 'For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward), self.k_reward_max)
        return a
        
    def run(self):
        # scale factor for luby sequence
        scale_factor = 8
        for i in range(self.iters):
            if LUBY:
                # calculating timeout following Luby sequence
                t = i
                if t == 0:
                    self.timeout[t] = 1
                elif t == self.iters-1:
                    self.timeout[t] = 0
                else:
                    kk = math.log2(t+1)
                    if kk - int(kk) < 0.0000001:
                        self.timeout[t] = scale_factor * max(int(math.pow(2, int(kk)-1)), 1)
                    else:
                        ind = int(t -  math.pow(2, int(kk)-1) + 1)
                        self.timeout[t] = scale_factor * max(self.timeout[ind], 1)
            else:
                if i == 0:
                    self.timeout[i] = scale_factor
                else:
                    self.timeout[i] = self.timeout[i-1] * 1.2 

            a = self.pull()
            self.reward[i] = self.mean_reward
            self.selections[a] += 1


            
    def reset(self, timeout = 0, states = []):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        self.k_reward_max = np.zeros(self.k)
        self.selections = np.zeros(self.k)
        if len(states) == 0:
            self.states = np.zeros(self.k)
        else:
            self.states = states
        self.timeout = np.zeros(self.iters)

def main(argv):
    
    ifile = ''
    propfile = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:", ["ifile=","propfile="])
    except getopt.GetoptError:
            print("MAB_eg_ABC.py  -i ifile")
            sys.exit(2)
            
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("MAB_eg_ABC.py -i ifile")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print("Input file is :" , ifile)


    # number of arms
    k = 3

    # number of steps/iterations/runs per episode
    iters = 500 #10

    eps_0_rewards = np.zeros(iters)
    eps_01_rewards = np.zeros(iters)
    eps_1_rewards = np.zeros(iters)

    eps_0_selection = np.zeros(k)
    eps_01_selection = np.zeros(k)
    eps_1_selection = np.zeros(k)

    # number of episodes, the reward value averaged over these episodes
    episodes = 1 #5

    eps_0_states = []
    eps_01_states = []
    eps_1_states = []


    # Initialize bandits with three different epsilon
    eps_0 = eps_bandit(k, 0.0, iters,T, inputfile)
    eps_01 = eps_bandit(k, 0.01, iters,T,  inputfile, eps_0.mu.copy())
    eps_1 = eps_bandit(k, 0.1, iters,T, inputfile, eps_0.mu.copy())
    
    # Run experiments
    for i in range(episodes):
        
        # Run experiments
        # eps_0.run()
        # eps_01.run()
        eps_1.run()
        
        # Update long-term averages
        eps_0_rewards = eps_0_rewards + (eps_0.reward - eps_0_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (eps_01.reward - eps_01_rewards) / (i + 1)
        eps_1_rewards = eps_1_rewards + (eps_1.reward - eps_1_rewards) / (i + 1)

        for j in range(k):
            eps_0_selection[j] += eps_0.selections[j]
            eps_01_selection[j] += eps_01.selections[j]
            eps_1_selection[j] += eps_1.selections[j]

        # restore state

        eps_0_states = eps_0.states.copy()
        eps_01_states = eps_01.states.copy()
        eps_1_states = eps_1.states.copy()

        # if i < episodes-1:
        #     t = 2*eps_0.timeout
        # else: # last run no timeout
        #     t = 0

        # timeout reset to 1s
        t = 1
        
        eps_0.reset(t, eps_0_states)
        eps_01.reset(t, eps_01_states)
        eps_1.reset(t, eps_1_states)
        
    fig1 = plt.figure(figsize=(12,8))
    plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
    plt.plot(eps_01_rewards, label="$\epsilon=0.01$")
    plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon-greedy$ Rewards after " + str(episodes) + " Episodes")
    plt.legend()
    # plt.show()

    bins = np.linspace(0, k-1, k)

    fig2 = plt.figure(figsize=(12,8))
    plt.bar(bins, eps_0_selection, width = 0.33, color='b',  label="$\epsilon=0$")
    plt.bar(bins+0.33, eps_01_selection, width=0.33, color='g', label="$\epsilon=0.01$")
    plt.bar(bins+0.66, eps_1_selection, width=0.33, color='r', label="$\epsilon=0.1$")
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlim([0,k])
    plt.title("Actions Selected by Each Algorithm")
    plt.xlabel("Action")
    plt.ylabel("Number of Actions Taken")
    # plt.show()

    opt_per = np.array([eps_0_selection, eps_01_selection, eps_1_selection]) / (episodes * iters) * 100
    df = pd.DataFrame(opt_per, index=['$\epsilon=0$', '$\epsilon=0.01$', '$\epsilon=0.1$'], columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)
    pp = PdfPages("plot_MAB_eg_ABC_all.pdf")
    # for fig in figs_ss:
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.close()

    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
