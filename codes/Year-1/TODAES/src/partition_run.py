import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
# from memory_profiler import memory_usage
from abcBMCUtil import *

PATH = "../"
DEBUG = True
DEBUG = False
OPT = True
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


Actions = ['bmc2', 'bmc3', 'bmc3s',  'bmc3g','bmc3r', 'bmcu', 'bmc3j','pdr']

def runseq(fn, seq):

	fname = os.path.join(PATH, fn)
	#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
	#ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	seq_list = ['({0}, {1})'.format(Actions[t[0]], t[1]) for t in seq]
	sequence = ';'.join(seq_list)
	print('##################################')
	print('---- Running sequence {0} ----'.format(sequence), fn, fname)
	
	#starting state
	sd = 0
	asrt = -1
	sm = None
	tt = 0
	pt = 0
	for item in seq:
		a, t = item
		if a == 0:    #ABC bmc2
			asrt, sm, ar_tab = bmc2(fname, sd, t)
		elif a == 1: #ABC bmc3
			asrt, sm, ar_tab = bmc3(fname, sd, t)
		elif a == 2: #ABC bmc3s
			asrt, sm, ar_tab = bmc3rs(fname, sd, t)
		elif a == 3: #ABC bmc3g
			asrt, sm, ar_tab = bmc3rg(fname, sd, t)
		elif a == 4: #ABC bmc3r
			asrt, sm, ar_tab = bmc3r(fname, sd, t)
		elif a == 5: #ABC bmc3u
			asrt, sm, ar_tab = bmc3ru(fname, sd, t)
		elif a == 6: #ABC bmc3j
			asrt, sm, ar_tab = bmc3j(fname, sd, t)
		elif a == 7: #ABC pdr
			asrt, sm, ar_tab = pdr(fname, t)

		if sm:
			sd = sm.frame+1 if sm.frame > 0 else sm.frame
		else:
			sm =  abc_result(frame=max(0, sd-1), conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = 0)
		print(sm)
		sys.stdout.flush()
		
		
		tt += sm.tt
		
		if asrt > 0:
			print('Output asserted at frame ', asrt, 'time', pt+sm.tt)
			print('Stopping iteration')
			pt += sm.tt
			
			break
		else:
			pt += t
		
	print('############### {0}: {1}, {2} ###################'.format( 'SAT' if asrt> 0 else 'TIMEOUT', sm.frame, pt if asrt > 0 else tt))
	return (asrt, 'SAT' if asrt> 0 else 'TIMEOUT', sm.frame, pt if asrt > 0 else tt)

def main(argv):
	
	inputfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:", ["ifile="])
	except getopt.GetoptError:
			print("partition_run.py  -i ifile")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("partition_run.py -i ifile")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
	print("Input file is :" , inputfile)

	k =  7 # arms
	TIMEOUT = 1800
	To1 = [120, 120, 240, 360, 480, 480]
	To2 = [60, 60, 60, 60, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120]
	#print('iters', iters, T)
		
	string = 'Engine & frames (cond) & time & time wasted & frames (cond) & time & time wasted & frames (cond) & time & time wasted\n'
	for i in range(k):
		seq = []
		seq.append((i, TIMEOUT))
		as1, cond1, sd1, tt1 = runseq(inputfile, seq)
		w1 = max(0, TIMEOUT - tt1)
		tot1 = tt1 if as1 > 0 else TIMEOUT
		
		seq = []
		for t in To1:
			seq.append((i, t))
		as2, cond2, sd2, tt2 = runseq(inputfile, seq)
		w2 = max(0, TIMEOUT - tt2)
		tot2 = tt2 if as2 > 0 else TIMEOUT
		
		seq = []
		for t in To2:
			seq.append((i, t))
		as3, cond3, sd3, tt3 = runseq(inputfile, seq)
		w3 = max(0, TIMEOUT - tt3)
		tot3 = tt3 if as3 > 0 else TIMEOUT
		
		string += '{0} & {1}({2}) & {3} & {4} & {5}({6}) & {7} & {8} & {9}({10}) & {11} & {12}\n'.format(Actions[i], sd1, cond1, tot1, w1, \
		sd2, cond2, tot2, w2, sd3, cond3, tot3, w3)	
	
	print('@@@@@@@@@@@@@@@@ RESULTS @@@@@@@@@@@@@@@@@')
	print(string)


	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
