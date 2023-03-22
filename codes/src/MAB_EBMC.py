import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time
from memory_profiler import memory_usage

PATH = "../"
DEBUG = True
# DEBUG = False

class bandit:

    def __init__(self, k, iters, fname = '',prop = ''):
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
        self.property = prop

    def get_reward(self, a):
        start_time = time.time()
        st = ''
        if a == 0:    #EBMC bmc
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--k-induction"] #, "--boound", "20"]
        elif a == 1:  #EBMC bdd aig
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--bdd"]
        elif a == 2: #EBMC bdd boolector
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--boolector"]

        elif a == 3: #EBMC mathsat    
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property),  "--mathsat"]

        elif a == 4: #EBMC z3    
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--z3"]
                
        if DEBUG:
            print(a,'\t----- '+str(st))

        # p = subprocess.run(st) #, capture_output=True)
        try:
            # gc.collect()
            output =  subprocess.check_output(st)#, timeout=6*3600)
            out = 0
        except subprocess.CalledProcessError as e:
            out = e.returncode
            output = e.stdout
        except Exception as e:
            print('Running call again....') 
            if DEBUG:
                print('\t----- '+str(st))
            try:
                output =  subprocess.check_output(st)
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
        
        res = 0
        if(out == 0):
            # if (a in [0,2,3,4]):
            if(b'SUCCESS' in output):
                res =  1/(end_time - start_time)
            # if(b'SAT' in output): b'UNSAT' in output or
            #     res = (end_time - start_time)
            # else:
            #     res = 0
            # elif a == 1:
            # if(b'SUCCESS' in output):
            #     return (end_time - start_time)
            if(b'UNKNOWN' in output):
                res = 0 # res
        return res
    
    def run(self):
        for i in range(self.iters):
            a = self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)


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
    
    def __init__(self, k, eps, iters, fname = '',prop = ''):
        bandit.__init__(self, k, iters, fname, prop)
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
    
    def __init__(self, k, iters, fname = '',prop = ''):
        bandit.__init__(self, k, iters, fname, prop)
       
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
    def __init__(self, k, c, iters, fname = '',prop = ''):
        bandit.__init__(self, k, iters, fname, prop)
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
    propfile = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:p:", ["ifile=","propfile="])
    except getopt.GetoptError:
            print("MAB_eps_greedy.py  -i ifile -p propfile")
            sys.exit(2)
            
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("MAB_eps_greedy.py -i ifile -p propfile")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-p", "--propfile"):
            propfile = arg
    print("Input file is :" , ifile, 'prop', propfile)


    fname = os.path.join(PATH, propfile)
    prop = ''
    with open(fname, "r") as f:
        prop = f.read()

    k = 5 # arms
    iters = 2000 # time-steps
    episodes = 10 #episodes

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
        eps_0 = eps_bandit(k, 0.0, iters,inputfile, prop)
        eps_01 = eps_bandit(k, 0.01, iters,  inputfile, prop)
        eps_1 = eps_bandit(k, 0.1, iters, inputfile, prop)
        eps_decay = eps_decay_bandit(k, iters,inputfile, prop)
        ucb1 = ucb1_bandit(k, 2, iters,inputfile, prop)

        
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
    pp = PdfPages("plot_MAB_all.pdf")
    # for fig in figs_ss:
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.close()

    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
