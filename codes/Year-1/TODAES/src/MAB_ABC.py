import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
# from memory_profiler import memory_usage

PATH = "../"
DEBUG = True
DEBUG = False
T = 60 

abc_result =  namedtuple('abc_result', ['frame', 'var', 'cla', 'conf', 'to']) 

class bandit:

    def __init__(self, k, iters, fname = ''):
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

        ##
        self.fname =  fname
        self.timeout = np.zeros(iters)


        # current state of the run; eg - bmc depth
        self.states = 0

    def get_reward(self, a):
        start_time = time.time()

        # get starting depth
        # sd = 0
        # if self.n  == 0:
        sd = int(self.states) #[a])

        pname = os.path.join(PATH, "ABC")
        cmdName = os.path.join(pname, "abc")
        fname = os.path.join(PATH, self.fname)
        st = ''

        t = int(self.timeout[self.n])
        if self.n == 0: #a == 3: #abc pdr    
            command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; \
            pdr -v -T {1:5d}; print_stats\"".format(fname, t)
            st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]'
        else:
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
        
            elif a == 3:  #ABC reach
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
        result = 0
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
            if self.n == 0:
                # elif (a == 3): # pdr
                xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
                m = re.finditer(xx, output, re.M|re.I)
                for m1 in m:
                    sm1 = int(m1.group(1)), float(m1.group(2))
                    if DEBUG:
                        print(sm1)  
                    sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, to=sm1[0]) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                res =  (end_time - start_time), sm
            else:

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
                    # result = sm.frame
                
                # elif (a == 3): # reach
                #     xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
                #     m = re.finditer(xx, output, re.M|re.I)
                #     for m1 in m:
                #         sm1 = int(m1.group(1)), float(m1.group(2))
                #         if DEBUG:
                #             print(sm1)
                #         sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, to=sm1[0]) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                #     res =  (end_time - start_time), sm
        if sm is not None:
            result = sm.frame
            self.states = sm.frame - 1 if sm.frame > 0 else sm.frame
        return result


    def run(self):
        for i in range(self.iters):
            if i == 0:
                self.timeout[i] = T
            else:
                self.timeout[i] = self.timeout[i-1] * 1.2 

            a = self.pull()
            self.reward[i] = self.mean_reward
            # self.timeout[i] = T
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
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
    
    def __init__(self, k, eps, iters, fname = ''):
        bandit.__init__(self, k, iters, fname)
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
        reward = self.get_reward(a)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

        # if a >0:
        print('For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
        return a
        
   
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
    
    def __init__(self, k, iters, fname = ''):
        bandit.__init__(self, k, iters, fname)
       
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
        res = self.get_reward(a)

        reward = res #max(0, 1/tm) ##-1*mem_use 
        # if DEBUG:
        # reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

        # if a >0:
        print('For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
        return a
       
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
    def __init__(self, k, c, iters, fname = ''):
        bandit.__init__(self, k, iters, fname)
        # Exploration parameter
        self.c = c
         
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n))
             
        reward = self.get_reward(a) #np.random.normal(self.mu[a], 1)
         
        # Update counts
        self.n += 1
        self.k_n[a] += 1
         
        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
         
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
    
def main(argv):
    
    ifile = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:", ["ifile="])
    except getopt.GetoptError:
            print("MAB_eps_greedy.py  -i ifile")
            sys.exit(2)
            
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("MAB_eps_greedy.py -i ifile")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print("Input file is :" , ifile)

    k = 2 # arms
    iters = 2000 # time-steps
    episodes = 2 #episodes

    eps_0_rewards = np.zeros(iters)
    eps_01_rewards = np.zeros(iters)
    eps_1_rewards = np.zeros(iters)
    eps_decay_rewards = np.zeros(iters)
    ucb1_rewards = np.zeros(iters)

    eps_0_selection = np.zeros(k)
    eps_01_selection = np.zeros(k)
    eps_1_selection = np.zeros(k)
    eps_decay_selection = np.zeros(k)
    ucb1_selection = np.zeros(k)

    # Run experiments
    for i in range(episodes):
        # Initialize bandits
        eps_0 = eps_bandit(k, 0.0, iters,inputfile)
        eps_01 = eps_bandit(k, 0.01, iters,  inputfile)
        eps_1 = eps_bandit(k, 0.1, iters, inputfile)
        eps_decay = eps_decay_bandit(k, iters,inputfile)
        ucb1 = ucb1_bandit(k, 2, iters,inputfile)

        
        # Run experiments
        eps_0.run()
        eps_01.run()
        eps_1.run()
        eps_decay.run()
        ucb1.run()
        
        # Update long-term averages
        eps_0_rewards = eps_0_rewards + (eps_0.reward - eps_0_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (eps_01.reward - eps_01_rewards) / (i + 1)
        eps_1_rewards = eps_1_rewards + (eps_1.reward - eps_1_rewards) / (i + 1)
        eps_decay_rewards = eps_decay_rewards + (eps_decay.reward - eps_decay_rewards) / (i + 1)
        ucb1_rewards = ucb1_rewards + (ucb1.reward - ucb1_rewards) / (i + 1)

        # Average actions per episode
        eps_0_selection = eps_0_selection + (eps_0.k_n - eps_0_selection) / (i + 1)
        eps_01_selection = eps_01_selection + (eps_01.k_n - eps_01_selection) / (i + 1)
        eps_1_selection = eps_1_selection + (eps_1.k_n - eps_1_selection) / (i + 1)
        eps_decay_selection = eps_decay_selection + (eps_decay.k_n - eps_decay_selection) / (i + 1)
        ucb1_selection = ucb1_selection + (ucb1.k_n - ucb1_selection) / (i + 1)
        
    fig1 = plt.figure(figsize=(12,8))
    plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
    plt.plot(eps_01_rewards, label="$\epsilon=0.01$")
    plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
    plt.plot(eps_decay_rewards, label="$\epsilon-decay$")
    plt.plot(ucb1_rewards, label="$ucb1$")
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards after " + str(episodes)  + " Episodes")
    plt.legend()
    # plt.show()

    bins = np.linspace(0, k-1, k)

    fig2 = plt.figure(figsize=(12,8))
    plt.bar(bins, eps_0_selection, width = 0.2, color='b',  label="$\epsilon=0$")
    plt.bar(bins+0.2, eps_01_selection, width=0.2, color='g', label="$\epsilon=0.01$")
    plt.bar(bins+0.4, eps_1_selection, width=0.2, color='r', label="$\epsilon=0.1$")
    plt.bar(bins+0.6, eps_decay_selection, width = 0.2, color='k',  label="$\epsilon-decay$")
    plt.bar(bins+0.8, ucb1_selection, width = 0.2, color='y',  label="$ucb1$")
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.xlim([0,k])
    plt.title("Actions Selected by Each Algorithm")
    plt.xlabel("Action")
    plt.ylabel("Number of Actions Taken")
    # plt.show()

    opt_per = np.array([eps_0_selection, eps_01_selection, eps_1_selection, eps_decay_selection, ucb1_selection]) / iters * 100
    df = pd.DataFrame(opt_per, index=['$\epsilon=0$', '$\epsilon=0.01$', '$\epsilon=0.1$', '$\epsilon-decay$', 'ucb1'], \
                        columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)
    pp = PdfPages("plot_MAB_ABC_all.pdf")
    # for fig in figs_ss:
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.close()

    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
