import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
# from memory_profiler import memory_usage

PATH = "../"
DEBUG = True
DEBUG = False
OPT = True
T = 30 
TIMEOUT = 3600
SC = 1.5
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
DIFF = 2 # function of depth, time, memory
DIFF = 3 # function average number of clauses/time
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
PDR = False

abc_result =  namedtuple('abc_result', ['frame', 'var', 'cla', 'conf', 'mem' ,'to', 'asrt']) 

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
        start_time = time.time()

        # get starting depth
        # sd = 0
        # if self.n  == 0:
        sd = int(self.states) #[a])

        pname = os.path.join(PATH, "ABC")
        cmdName = os.path.join(pname, "abc")
        fname = os.path.join(PATH, self.fname)
        #ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
        ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))
        if DEBUG or self.n == 0:
            print(fname, ofname)

        st = ''

        t = int(self.timeout[self.n])
        if PDR and self.n == 0: #a == 3: #abc pdr    
            command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {2};&get; \
            pdr -v -T {1:5d}; print_stats\"".format(fname, t, ofname)
            st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]'
        else:
            if a == 0:    #ABC bmc2
                if self.n == 0:
                    command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {3};&get; \
                    bmc2 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(fname, sd, t, ofname)
                else:
                    command = "\"read {0}; print_stats; &get; bmc2 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(ofname, sd, t)
                st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]

            elif a == 1: #ABC bmc3
                if self.n == 0:
                    command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {3};&get; \
                    bmc3 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(fname, sd, t, ofname)
                else:
                    command = "\"read {0}; print_stats; &get; bmc3 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(ofname, sd, t)
                st = ' '.join([cmdName, "-c", command])  #, "--boound", "20"]

            elif a == 2: #ABC &bmc
                if self.n == 0:
                    command = "\"read {0};  print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {3};&get; \
                    &bmc -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(fname, sd, t, ofname)
                else:
                    command = "\"read {0}; print_stats; &get; &bmc -S {1} -T 10 -v; print_stats\"".format(ofname, sd)
                st = ' '.join([cmdName, "-c", command])  #, "--boound", "20"]
        
            elif a == 3:  #ABC reach
                if self.n == 0:
                    command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {2};&get; \
                    reach -T {1:5d} -F 1 -v -L stdout; print_stats\"".format(fname, t, ofname)
                else:
                    command = "\"read {0}; print_stats; &get; reach -T {1:5d} -F 1 -v -L stdout; print_stats\"".format(ofname, t)
                st = ' '.join([cmdName, "-c", command])  #, "--boound", "20"]
        
        #if DEBUG:
        #   print(a,'\t '+str(st))

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
            print('Cmd res:', out) #, output)
        
        end_time = time.time()

        '''This will give you the output of the command being executed'''
        if DEBUG:
            print ("\t----- Output: " + str(out)) #,  'out', output)       
        
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
            if PDR and self.n == 0:
                # elif (a == 3): # pdr
                xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
                m = re.finditer(xx, output, re.M|re.I)
                xx2 = r'Output[ \t]+([\d]+)[.]+[ \t]+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+)'
                m3 = re.finditer(xx2, output, re.M|re.I)
                asrt = -1
                for m31 in m3:
                    if DEBUG:
                        print(m31.group(2)) 
                    asrt = m31.group(2)
                for m1 in m:
                    sm1 = int(m1.group(1)), float(m1.group(2))
                    if DEBUG:
                        print(sm1)  
                    sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, mem = -1, to=sm1[0], asrt=asrt) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                res =  (end_time - start_time), sm
            else:

                if (a == 0): #bmc2
                    xx = r'[ \t]+([\d]+)[ \t]+[:][ \t]+F[ \t]+=[ \t]+([\d]+).[ \t]+O[ \t]+=[ \t]+([\d]+).[ \t]+And[ \t]+=[ \t]+([\d]+).[ \t]+Var[ \t]+=[ \t]+([\d]+).[ \t]+Conf[ \t]+=[ \t]+([\d]+).[ \t]+([\d]+)[ \t]+MB[ \t]+([\d]+[.][\d]+)[ \t]+sec'
                    m = re.finditer(xx, output, re.M|re.I)
                    if DEBUG:
                        print(m)
                    xx1 = r'No[ \t]+output[ \t]+failed[ \t]+in[ \t]+([\d]+)[ \t]+frames[.]*'
                    m2 = re.finditer(xx1, output, re.M|re.I)
                    for m21 in m2:
                        if DEBUG:
                            print(m21.group(1)) 
                    xx2 = r'Output[ \t]+([\d]+)[.]+[ \t]+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+)'
                    m3 = re.finditer(xx2, output, re.M|re.I)
                    asrt = -1
                    for m31 in m3:
                        if DEBUG:
                            print(m31.group(2)) 
                        asrt = m31.group(2)
                    for m1 in m:
                        sm1 = int(m1.group(2)), int(m1.group(5)), int(m1.group(4)), int(m1.group(6)), int(m1.group(7)), float(m1.group(8))
                        if DEBUG:
                            print(sm1, m1.group(1), m21.group(1), asrt)      
                        sm = abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[4], to=sm1[5], asrt=asrt) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                    res =  (end_time - start_time), asrt, sm

                elif (a == 1): #bmc3
                    xx = r'[ \t]*([\d]+)[ \t]+[+][ \t]+[:][ \t]+Var[ \t]+=[ \t]+([\d]+).[ \t]+Cla[ \t]+=[ \t]+([\d]+).[ \t]+Conf[ \t]+=[ \t]+([\d]+).[ \t]+Learn[ \t]+=[ \t]+([\d]+).[ \t]+[\d]+[ \t]+MB[ \t]+([\d]+)[ \t]+MB[ \t]+([\d]+[.][\d]+)[ \t]+sec'
                    m = re.finditer(xx, output, re.M|re.I)
                    if DEBUG:
                        print(m)
                    xx2 = r'Output[ \t]+([\d]+)[.]+[ \t]+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+)'
                    m3 = re.finditer(xx2, output, re.M|re.I)
                    asrt = -1
                    for m31 in m3:
                        if DEBUG:
                            print(m31.group(2)) 
                        asrt = m31.group(2)
                    for m1 in m:
                        sm1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3)), int(m1.group(4)), int(m1.group(5)), int(m1.group(6)),float(m1.group(7))
                        if DEBUG:
                            print(sm1)   
                        sm =  abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[5],to=sm1[6], asrt=asrt) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                    res =  (end_time - start_time), asrt, sm

                elif (a == 2): # &bmc
                    # print(output)
                    xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
                    m = re.finditer(xx, output, re.M|re.I)
                    # print(m)
                    for m1 in m:
                        sm1 = int(m1.group(1)), float(m1.group(2))
                        # print(sm1)   
                        sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, mem = -1,to=sm1[0])  #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
                        #print(sm)
                    sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, mem = -1, to=sm1[0])
                    res =  (end_time - start_time), asrt, sm
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
            reward = 0
            if DIFF == 0 :
                reward = (sm.frame - sd)
            elif DIFF == 1:
                reward = sm.frame
            elif DIFF == 2:
                reward =  np.exp(0.3*sm.frame - 0.2*sm.mem - 0.1*sm.to/60)
            else:
                reward = sm.cla/sm.to if sm.to > 0 else sm.to
            self.states = sm.frame+1 if sm.frame > 0 else sm.frame
        else:
            sm =  abc_result(frame=sd-1, conf=0, var=0, cla=0, mem = -1, to=to, asrt = asrt)
            if asrt > 0:
                reward = asrt
            else:
                reward = 0
        return reward, sm


    def run(self):
        totalTime = 0
        seq = []
        for i in range(self.iters):
            if totalTime > TIMEOUT:
                print('BMC-depth reached ', self.states)
                print('Stopping iteration')
                break
            if i == 0:
                self.timeout[i] = T
            else:
                self.timeout[i] = self.timeout[i-1] * SC 

            a, reward, sm = self.pull()
            if sm:
                seq.append((a, reward, self.states, sm.to))
            else:
                seq.append((a, reward, self.states, -1))
            self.reward[i] = self.mean_reward

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

    k = 2 # arms
    #iters = int((TIMEOUT/T)) 
    iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
    episodes = 1 #episodes
    print('iters', iters)
    alpha = 0.1
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
    labels = [r'$\epsilon=0$', r'$\epsilon=0.01$', r'$\epsilon=0.1$', r'$\epsilon=0.1, \alpha = 0.1$', r'$\epsilon-decay$', 'ucb1', \
    r'opt $\epsilon=0$', r'opt $\epsilon=0.01$', r'opt $\epsilon=0.1$', r'opt $\epsilon=0.1, \alpha = 0.1$', \
        r'opt $\epsilon-decay$', r'opt ucb1']

    fname = (inputfile.split('/')[-1]).split('.')[0]
    print(fname)
    pp = PdfPages("plot_MAB_BMC_all_{0}_{1}{2}.pdf".format(fname,T, '_DIFF' if DIFF else ''))
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
    df = pd.DataFrame(opt_per, index=labels, \
                            columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)
    
    pp.close()


    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
