import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time
from memory_profiler import memory_usage

PATH = "../"
DEBUG = True
DEBUG = False

class bandit:

    def __init__(self, k, iters, alpha = 1, init = 0, fname = '',prop = '', top = ''):
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

        # recency constant
        self.alpha = alpha

        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        for i in range(k):
            self.k_reward[i] = init

        ##
        self.fname =  fname
        self.property = prop
        self.top = top

    def get_reward(self, a):
        start_time = time.time()
        st = ''
        if a == 0:    #EBMC bound 
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--bound", "20"]
            if len(str(self.property)) == 0:
                st = [cmdName, fname, "--top", str(self.top), "--bound", "20"]

        elif a == 1:  #EBMC bdd aig
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--bdd"]
            if len(str(self.property)) == 0:
                st = [cmdName, fname, "--top", str(self.top),  "--bdd"]

        elif a == 2: #EBMC k-induction
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--k-induction", "--bound", "20"]
            if len(str(self.property)) == 0:
                st = [cmdName, fname, "--top", str(self.top), "--k-induction", "--bound", "20"]

        elif a == 3: #EBMC mathsat    
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--mathsat", "--bound 20"]
            if len(str(self.property)) == 0:
                st = [cmdName, fname, "--top", str(self.top), "--mathsat", "--bound 20"]

        elif a == 4: #EBMC z3    
            pname = os.path.join(PATH, "EBMC")
            cmdName = os.path.join(pname, "ebmc")
            fname = os.path.join(PATH, self.fname)
            st = [cmdName, fname, "-p", str(self.property), "--z3", "--bound 20"]
            if len(str(self.property)) == 0:
                st = [cmdName, fname, "--top", str(self.top),"--z3", "--bound 20"]
                
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
                # res =  np.exp(-0.5* (end_time - start_time))
            # if(b'SAT' in output): b'UNSAT' in output or
            #     res = (end_time - start_time)
            # else:
            #     res = 0
            # elif a == 1:
            # if(b'SUCCESS' in output):
            #     return (end_time - start_time)
            if(b'UNKNOWN' in output) or (b'FAILURE' in output) :
                res = 0 # res
        else:
            res = -2
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
    
    def __init__(self, k, eps, iters, alpha= 1, init = 0, fname = '',prop = '', top = ''):
        bandit.__init__(self, k, iters, alpha, init, fname, prop, top)
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
        if self.alpha == 1:
            self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
        else:
            self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) * self.alpha

        # if a >0:
        print(self.n, 'For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
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
    
    def __init__(self, k, iters,  alpha= 1, init = 0, fname = '',prop = '', top = ''):
        bandit.__init__(self, k, iters, alpha, init, fname, prop, top)
        # Search probability
       
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
        if self.alpha == 1:
            self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
        else:
            self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) * self.alpha

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
    def __init__(self, k, c, iters, alpha= 1, init = 0, fname = '',prop = '', top = ''):
        bandit.__init__(self, k, iters, alpha, init, fname, prop, top)
        # Search probability
        # Exploration parameter
        self.c = c
        self.k_ucb_reward = np.zeros(self.k)
         
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_ucb_reward)
        # a = np.argmax(self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n))
             
        reward = self.get_reward(a) #np.random.normal(self.mu[a], 1)
         
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
        print('For action {0} reward {1}, updated reward {2}-{3}'.format(a, reward, self.k_reward, self.k_ucb_reward), self.c *np.sqrt((np.log(self.n)) / self.k_n))
    
def main(argv):
    
    ifile = ''
    propfile = ''
    top = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:p:t:", ["ifile=","propfile=","top="])
    except getopt.GetoptError:
            print("MAB_eps_greedy.py  -i ifile -p propfile -t top")
            sys.exit(2)
            
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("MAB_eps_greedy.py -i ifile -p propfile -t top")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-p", "--propfile"):
            propfile = arg
        elif opt in ("-t", "--top"):
            top = arg
    print("Input file is :" , ifile, 'prop', propfile, 'top', top)

    prop = ''
    if len(propfile) > 0:
        fname = os.path.join(PATH, propfile)
        with open(fname, "r") as f:
            prop = f.read()

    k = 5 # arms
    iters = 200 # time-steps
    episodes = 5 #episodes
    # opt_init = 10,000

    print('iters', iters)
    alpha = 0.05
    c = 200

    # o_eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile, prop, top)
    # o_eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile, prop, top)
    # o_eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile, prop, top)
    # o_eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile, prop, top)
    # o_ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile, prop, top)

    # eps_0 = eps_bandit(k, 0.0, iters, 0, inputfile, prop, top)
    # eps_01 = eps_bandit(k, 0.01, iters, 0, inputfile, prop, top)
    # eps_1 = eps_bandit(k, 0.1, iters, 0, inputfile, prop, top)
    # eps_decay = eps_decay_bandit(k, 0, iters,inputfile, prop, top)
    # ucb1 = ucb1_bandit(k, 2, iters, 0, inputfile, prop, top)
    # opt_eps = eps_bandit(k, 0.01, iters, opt_init, inputfile, prop, top)

    eps_0_rewards = np.zeros(iters)
    eps_01_rewards = np.zeros(iters)
    eps_1_rewards = np.zeros(iters)
    eps_decay_rewards = np.zeros(iters)
    ucb1_rewards = np.zeros(iters)
    opt_eps_rewards = np.zeros(iters)
    eps_1_alpha_rewards = np.zeros(iters)


    eps_0_selection = np.zeros(k)
    eps_01_selection = np.zeros(k)
    eps_1_selection = np.zeros(k)
    eps_decay_selection = np.zeros(k)
    ucb1_selection = np.zeros(k)
    opt_eps_selection = np.zeros(k)
    eps_1_alpha_selection = np.zeros(k)


    # options = [eps_0, eps_01, eps_1, eps_decay, ucb1, o_eps_01, eps_1_alpha]
    labels = [r'$\epsilon=0$', r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon-decay$', 'ucb1', r'opt $\epsilon=0.01$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha) ]

    # Run experiments
    for i in range(episodes):
        print('---------- Episodes {0} ---------'.format(i))
        # Initialize bandits
        reward = 0
        eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile, prop, top)
        eps_01 = eps_bandit(k, 0.01, iters, 1, reward,  inputfile, prop, top)
        eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile, prop, top)
        eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile, prop, top)
        ucb1 = ucb1_bandit(k, c, iters,1,  reward, inputfile, prop, top)
        eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile, prop, top)

        reward = 10000
        opt_eps = eps_bandit(k, 0.01, iters, 1, reward,  inputfile, prop, top)

        # Run experiments
        # eps_0.run()
        # eps_01.run()
        # eps_1.run()
        # eps_decay.run()
        ucb1.run()
        # opt_eps.run()
        # eps_1_alpha.run()
        
        # Update long-term averages
        eps_0_rewards = eps_0_rewards + (eps_0.reward - eps_0_rewards) / (i + 1)
        eps_01_rewards = eps_01_rewards + (eps_01.reward - eps_01_rewards) / (i + 1)
        eps_1_rewards = eps_1_rewards + (eps_1.reward - eps_1_rewards) / (i + 1)
        eps_decay_rewards = eps_decay_rewards + (eps_decay.reward - eps_decay_rewards) / (i + 1)
        ucb1_rewards = ucb1_rewards + (ucb1.reward - ucb1_rewards) / (i + 1)
        opt_eps_rewards = opt_eps_rewards + (opt_eps.reward - opt_eps_rewards) / (i + 1)
        eps_1_alpha_rewards = eps_1_alpha_rewards + (eps_1_alpha.reward - eps_1_alpha_rewards) / (i + 1)


        # Average actions per episode
        eps_0_selection = eps_0_selection + (eps_0.k_n - eps_0_selection) / (i + 1)
        eps_01_selection = eps_01_selection + (eps_01.k_n - eps_01_selection) / (i + 1)
        eps_1_selection = eps_1_selection + (eps_1.k_n - eps_1_selection) / (i + 1)
        eps_decay_selection = eps_decay_selection + (eps_decay.k_n - eps_decay_selection) / (i + 1)
        ucb1_selection = ucb1_selection + (ucb1.k_n - ucb1_selection) / (i + 1)
        opt_eps_selection  = opt_eps_selection  + (opt_eps.k_n - opt_eps_selection) / (i + 1)
        eps_1_alpha_selection  = eps_1_alpha_selection  + (eps_1_alpha.k_n - eps_1_alpha_selection) / (i + 1)
        
    fig1 = plt.figure(figsize=(16,12))
    plt.plot(eps_0_rewards, label=r"$\epsilon=0$ (greedy)")
    plt.plot(eps_01_rewards, label=r"$\epsilon=0.01$")
    plt.plot(eps_1_rewards, label=r"$\epsilon=0.1$")
    plt.plot(eps_decay_rewards, label=r"$\epsilon-decay$")
    plt.plot(ucb1_rewards, label="$ucb1$")
    plt.plot(opt_eps_rewards, label= r'opt $\epsilon=0.01$')
    plt.plot(eps_1_alpha_rewards,  label=r'$\epsilon=0.1, \alpha = {0}$'.format(alpha) )
    plt.legend() #bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards after " + str(episodes)  + " Episodes")
    plt.legend()
    # plt.show()

    bins = np.linspace(0, k-1, k)

    w = 1/7.0
    fig2 = plt.figure(figsize=(16,8))
    plt.bar(bins, eps_0_selection, width = w, color='b',  label=r"$\epsilon=0$")
    plt.bar(bins+w, eps_01_selection, width=w, color='g', label=r"$\epsilon=0.01$")
    plt.bar(bins+2*w, eps_1_selection, width=w, color='r', label=r"$\epsilon=0.1$")
    plt.bar(bins+3*w, eps_decay_selection, width = w, color='k',  label=r"$\epsilon-decay$")
    plt.bar(bins+4*w, ucb1_selection, width =w, color='c',  label="$ucb1$")
    plt.bar(bins+5*w, opt_eps_selection, width = w, color='y',  label=r'opt $\epsilon=0.01$')
    plt.bar(bins+6*w, eps_1_alpha_selection, width =w, color='m',  label=r'$\epsilon=0.1, \alpha = {0}$'.format(alpha) )
    plt.legend() #bbox_to_anchor=(1.2, 0.5))
    plt.xlim([0,k+1])
    plt.title("Actions Selected by Each Algorithm")
    plt.xlabel("Action")
    plt.ylabel("Number of Actions Taken")
    # plt.show()

    opt_per = np.array([eps_0_selection, eps_01_selection, eps_1_selection, eps_decay_selection, ucb1_selection, opt_eps_selection, eps_1_alpha_selection]) / iters * 100
    df = pd.DataFrame(opt_per, index=[r'$\epsilon=0$', r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon-decay$', 'ucb1',  r'opt $\epsilon=0.01$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha) ], \
                        columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)

    fname = (inputfile.split('/')[-1]).split('.')[0]
    print(fname)
    pp = PdfPages("plot_MAB_EBMC_{0}.pdf".format(fname))
    # for fig in figs_ss:
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.close()

    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
