import numpy as np 
import matplotlib.pyplot as plt 

import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math, csv
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
MAX_CLAUSE = 1e9
MAX_TIME = 3600
MAX_MEM = 4000
TIMEOUT = 3600

ST1 = 120
ST2 = 300

k =  4 # arms

Actions = ['bmc2', 'bmc3', 'bmc3s', 'bmc3g', 'bmc3r', 'bmcu', 'bmc3j','pdr']

def get_fname(fn):
	fname = os.path.join(PATH, fn)
	# ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))	
	ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	#if DEBUG or self.n == 0:
	print(fname, ofname)

	# if self.n == 0:
	simplify(fname, ofname)
	print('Simplified model', ofname)
	return ofname

def runseq(fname, seq):

	#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
	#ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	seq_list = ['({0}, {1})'.format(Actions[t[0]], t[1]) for t in seq]
	sequence = ';'.join(seq_list)
	print('##################################')
	print('---- Running sequence {0} ----'.format(sequence), fname)
	ofname = fname

	#starting state
	sd = 0
	asrt = -1
	sm = None
	tt = 0
	pt = 0
	ar_tab = {}
	for item in seq:
		a, t = item
		if a == 0:    #ABC bmc2
			asrt, sm, ar_tab1, tt1 = bmc2(ofname, sd, t)
		elif a == 1: #ABC bmc3
			asrt, sm, ar_tab1, tt1 = bmc3(ofname, sd, t)
		elif a == 2: #ABC bmc3s
			asrt, sm, ar_tab1, tt1 = bmc3rs(ofname, sd, t)
		elif a == 3: #ABC bmc3g
			asrt, sm, ar_tab1, tt1 = bmc3rg(ofname, sd, t)
		elif a == 4: #ABC bmc3r
			asrt, sm, ar_tab1, tt1 = bmc3r(ofname, sd, t)
		elif a == 5: #ABC bmc3u
			asrt, sm, ar_tab1, tt1 = bmc3ru(ofname, sd, t)
		elif a == 6: #ABC bmc3j
			asrt, sm, ar_tab1, tt1 = bmc3j(ofname, sd, t)
		elif a == 7: #ABC pdr
			asrt, sm, ar_tab1, tt1 = pdr(ofname, t)

		for ky in ar_tab1:
			ar_tab.update({ky: ar_tab1[ky]})

		if sm:
			sd = sm.frame+1 if sm.frame > 0 else sm.frame
			# if a == 0:
			# 	sd = sm.frame #if sm.frame > 0 else sm.frame
		else:
			sm =  abc_result(frame=max(0, sd-1), conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = tt1, ld = max(0, sd-1))
		print(sm)
		sys.stdout.flush()
		
		
		tt += sm.tt
		
		if asrt > 0:
			pt += sm.tt
			print('Output asserted at frame ', asrt, 'time', pt)
			print('Stopping iteration')
			
			break
		else:
			pt += t
	
	frame = asrt if asrt > 0  else sm.frame
	print('############### {0}: {1}, {2} ###################'.format( 'SAT' if asrt> 0 else 'TIMEOUT', asrt if asrt > 0  else sm.frame, pt if asrt > 0 else tt))
	return (asrt, 'SAT' if asrt> 0 else 'TIMEOUT', frame, pt if asrt > 0 else tt, ar_tab)

def get_reward(asrt, frames, clauses, mems, times ):
	rewards = [[],[],[]]
	cu_rewards = [[],[],[]]
	cu_re1 = 0
	cu_re2 = 0
	cu_re3 = 0
	c1 = 1.0/MAX_MEM
	c2 = 1.0/MAX_TIME
	for j in range(len(frames)):
		frame = frames[j]
		cla = clauses[j]
		mem = mems[j]
		to = times[j]
		# reward =  np.exp(0.3*frame/MAX_FRAME + 0.2*cla/MAX_CLAUSE - 0.5*to/MAX_TIME)
		# if asrt > 0:
		# 	reward =  np.exp(1 - 0.2*frame/MAX_FRAME - 0.3*to/MAX_TIME)
		#reward = 2*np.exp(-1*to/(1+frame)) 

		reward1 = 2*np.exp(-to/(1+frame)) 
		cu_re1 += reward1
		rewards[0].append(reward1)
		cu_rewards[0].append(cu_re1/(j+1))

		reward2 = 2*np.exp(-c1*mem/(1+frame)) 
		cu_re2 += reward2
		rewards[2].append(reward2)
		cu_rewards[2].append(cu_re2/(j+1))

		reward3 = to/(1+frame) 
		cu_re3 += reward3
		rewards[1].append(cu_re3)
		cu_rewards[1].append(np.exp(-1*cu_re3/(j+1)))

	return rewards, cu_rewards

def part_res(a, fname, ofname, TO, name, part):
	seq = []
	for to in TO:
		seq.append((a, to))

	as1, cond1, sd1, tt1, ar_tab1 = runseq(ofname, seq)
	w1 = max(0, TIMEOUT - tt1)
	tot1 = tt1 if as1 > 0 else TIMEOUT

	string = '\t {0}  {1:0.2f}  {2:0.2f}\t'.format( sd1, tot1, w1) 
	frames1, clauses1, mems1, times1 = [ar_tab1[ky].frame for ky in ar_tab1.keys()], [ar_tab1[ky].cla for ky in ar_tab1.keys()], [ar_tab1[ky].mem for ky in ar_tab1.keys()], [ar_tab1[ky].to for ky in ar_tab1.keys()]
	rewards1, cu_rewards1 = get_reward(as1, frames1, clauses1, mems1, times1)
	part1 = frames1, clauses1, mems1, times1, rewards1, cu_rewards1

	rows1 = []
	# rows1.append(header)
	for kk in range(len(frames1)):
		row = []
		row.append(Actions[a])
		row.append(fname+str(name))
		row.append(frames1[kk])
		row.append(clauses1[kk])
		row.append(mems1[kk])
		row.append(times1[kk])
		row.append(rewards1[0][kk])
		row.append(cu_rewards1[0][kk])
		row.append(rewards1[1][kk])
		row.append(cu_rewards1[1][kk])
		row.append(rewards1[2][kk])
		row.append(cu_rewards1[2][kk])
		rows1.append(row)

	return string, part1, rows1

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

	fname = (inputfile.split('/')[-1]).split('.')[0]
	# print(fname)

	ofname = get_fname(inputfile)

	filename = "plots_IF/BMC_records_{0}_{1}.csv".format(TIMEOUT, fname)
	# header = ['Design', 'Frame', 'Clauses', 'Mem', 'time']
	# writing to csv file 
	with open(filename, 'w+') as csvfile: 
		print('filename', fname)


	# pdfname = "plots/plot_Partition_{0}_{1}.pdf".format(fname, TIMEOUT)

	To = [TIMEOUT]
	K = [0, 3]
	
	header = ['Design', 'Action',  'Frame', 'Clauses', 'Mem', 'time', 'reward']
	figs = []
	string = 'Design Engine depth total ITF\n'
	for i in K:

		string += '{0} \t {1} \t'.format(fname, Actions[i]) 
		tr1, part1, rows1 = part_res(i, fname, ofname, To, 'Total', 1)
		frames1, clauses1, mems1, times1, rewards1, cu_rewards1 = part1
		string += tr1 + '\n'
		

		filename = "plots_IF/BMC_records_{0}_{1}.csv".format(TIMEOUT, fname)
		# # header = ['Design', 'Frame', 'Clauses', 'Mem', 'time']
		# # writing to csv file 
		# with open(filename, 'a+') as csvfile: 
		#     # creating a csv writer object 
		#     csvwriter = csv.writer(csvfile) 
		        
		#     # writing the fields 
		#     csvwriter.writerow(header) 
		#     csvwriter.writerows(rows1)

	print('@@@@@@@@@@@@@@@@ RESULTS @@@@@@@@@@@@@@@@@')
	print(string)


	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')
