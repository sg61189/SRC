
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple, OrderedDict


DEBUG = True
DEBUG = False

PATH = "../"
abc_result =  namedtuple('abc_result', ['frame', 'var', 'cla', 'conf', 'mem' ,'to', 'asrt']) 

# output = 'Output 0 of miter "../benchmark/HWMCC15/6s20_n" was asserted in frame 9. Time =    30.04 sec'
# print(m3, asrt)
# exit()

def run_cmd(command):
    start_time = time.time()
    try:
        # gc.collect()
        output =  subprocess.run(command, shell=True, capture_output = True, text = True).stdout#.strip("\n")
        #subprocess.check_output(command)#, timeout=6*3600)
        out = 0
    except subprocess.CalledProcessError as e:
        out = e.returncode
        output = e.stdout
    except Exception as e:
        print('Running call again....') 
        if DEBUG:
            print('\t----- '+str(command))
        try:
            output =  subprocess.run(command, shell=True, capture_output = True, text = True).stdout#.strip("\n") #subprocess.check_output(st)
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
    print('\t', output)

    sys.stdout.flush()
    return out, output

def simplify(fname, ofname):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {1}\"".format(fname, ofname)
    st = ' '.join([cmdName, "-c", command])
    out, output =  run_cmd(st)


def parse_bmc2(output):
    ar_tab = OrderedDict()
    sm = None
    xx = r'[ \t]+([\d]+)[ \t]+[:][ \t]+F[ \t]+=[ \t]*([\d]+).[ \t]+O[ \t]+=[ \t]*([\d]+).[ \t]+And[ \t]+=[ \t]*([\d]+).[ \t]+Var[ \t]+=[ \t]*([\d]+).[ \t]+Conf[ \t]+=[ \t]*([\d]+).[ \t]+.*([\d]+)[ \t]+MB[ \t]+([\d]+[.][\d]+)[ \t]+sec'
    m = re.finditer(xx, output, re.M|re.I)
    if DEBUG:
        print(m)
    xx1 = r'No[ \t]+output[ \t]+failed[ \t]+in[ \t]+([\d]+)[ \t]+frames[.]*'
    m2 = re.finditer(xx1, output, re.M|re.I)
    for m21 in m2:
        if DEBUG:
            print(m21.group(1)) 
    #Output 0 of miter "../benchmark/HWMCC15/6s20_n" was asserted in frame 9
    xx2 = r'Output.+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+).[.]*'
    m3 = re.finditer(xx2, output, re.M|re.I)
    asrt = -1
    for m31 in m3:
        if DEBUG:
            print(m31.group(1)) 
        asrt = int(m31.group(1))
    pretm = 0
    for m1 in m:
        sm1 = int(m1.group(2)), int(m1.group(5)), int(m1.group(4)), int(m1.group(6)), int(m1.group(7)), float(m1.group(8))
        if DEBUG:
            print(sm1, m1.group(1), m21.group(1), asrt)      
        sm = abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[4], to=sm1[5] - pretm, asrt=asrt) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
        pretm = sm1[5]
        ar_tab.update({sm.frame:sm})  
    sm_res = ar_tab[next(reversed(ar_tab))]  
    res =  asrt, sm_res
    return res

def parse_bmc3(output):
    ar_tab = OrderedDict()
    sm = None
    xx = r'[ \t]*([\d]+)[ \t]+[+-][ \t]+[:][ \t]+Var[ \t]+=[ \t]*([\d]+).[ \t]+Cla[ \t]+=[ \t]*([\d]+).[ \t]+Conf[ \t]+=[ \t]*([\d]+).[ \t]+Learn[ \t]+=[ \t]*([\d]+).[ \t]+.*[\d]+[ \t]+MB[ \t]+([\d]+)[ \t]+MB[ \t]+([\d]+[.][\d]+)[ \t]+sec'
    m = re.finditer(xx, output, re.M|re.I)
    if DEBUG:
        print(m)
    #Output 0 of miter "../benchmark/HWMCC15/6s20_n" was asserted in frame 9
    #xx2 = r'Output[.]*was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+).[.]*'
    #xx2 = r'Output[ \t]+([\d]+)[.]+[ \t]+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+)'
    xx2 = r'Output.+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+).[.]*'
    m3 = re.finditer(xx2, output, re.M|re.I)
    asrt = -1
    for m31 in m3:
        if DEBUG:
            print(m31.group(1)) 
        asrt = int(m31.group(1))
    pretm = 0
    for m1 in m:
        sm1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3)), int(m1.group(4)), int(m1.group(5)), int(m1.group(6)),float(m1.group(7))
        #tm = float(m1.group(7))
        if DEBUG:
            print(sm1)   
        sm =  abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[5],to=sm1[6]-pretm, asrt=asrt) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
        pretm = sm1[6]
        ar_tab.update({sm.frame:sm})  
    sm_res = ar_tab[next(reversed(ar_tab))] 
    res =  asrt, sm_res
    return res

def pdr(ofname, t):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; pdr -v -T {1:5d}; print_stats\"".format(ofname, t)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]'
    out, output =  run_cmd(st)

    sm = None
    ar_tab = OrderedDict()
    xx = r'Reached[ \t]+timeout[ \t]+[(]([\d]+)[ \t]+seconds[)][ \t]+in[ \t]+frame[ \t]+([\d]+)[.]*'
    m = re.finditer(xx, output, re.M|re.I)
    #xx2 = r'Output[ \t]+([\d]+)[.]+[ \t]+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+)'
    xx2 = r'Output.+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+).[.]*'
    m3 = re.finditer(xx2, output, re.M|re.I)
    asrt = -1
    for m31 in m3:
        if DEBUG:
            print(m31.group(1)) 
        asrt = m31.group(1)
    for m1 in m:
        sm1 = int(m1.group(1)), float(m1.group(2))
        if DEBUG:
            print(sm1)  
        sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, mem = -1, to=sm1[0], asrt=asrt) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
        ar_tab.update({sm.frame:sm})  
    sm_res = ar_tab[next(reversed(ar_tab))] 
    res =  asrt, sm_res
    return res

def bmc2(ofname, sd, t):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc2 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(ofname, (sd-1) if sd > 1 else 0, t)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    out, output =  run_cmd(st)
    res = parse_bmc2(output)
    return res

def bmc3(ofname, sd, t):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(ofname, sd, t)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    out, output =  run_cmd(st)
    res = parse_bmc3(output)
    return res

def bmc3s(ofname, sd, t):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -rs -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(ofname, sd, t)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    out, output =  run_cmd(st)
    res = parse_bmc3(output)
    return res

def bmc3j(ofname, sd, t, c = 5000, d = 5000, j = 2):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command1 = "\"read {0}; print_stats; &get; bmc3 -r -C {1:5d} -D {2:5d} -S {3:5d} -T {4:5d} -J {5} -v -L stdout; print_stats\"".format(ofname, c, d, sd, int(t/2), j)
    #command = "\"read {0}; print_stats; &get; bmc3 -r -C {1:5d} -D {2:5d} -S {3:5d} -T {4:5d} -v -L stdout; print_stats\"".format(ofname, c, d, sd, t,)
    st1 = ' '.join([cmdName, "-c", command1]) #, "--boound", "20"]
    out, output1 =  run_cmd(st1)
    command2 = "\"read {0}; print_stats; &get; bmc3 -r -C {1:5d} -D {2:5d} -S {3:5d} -T {4:5d} -J {5} -v -L stdout; print_stats\"".format(ofname, c, d, sd+1, int(t/2), j)
    st2 = ' '.join([cmdName, "-c", command2]) #, "--boound", "20"]
    out, output2 =  run_cmd(st2)
    res = parse_bmc3('\n'.join([output1,output2]))
    #sm =  abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[5],to=sm1[6]-pretm, asrt=asrt)
    return res

def bmc3az(ofname, sd, t, g = 10, h = 0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -azr -G {1:5d} -H {2:5d} -S {3:5d} -T {4:5d} -v -L stdout; print_stats\"".format(ofname, g, h, sd, t)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    out, output =  run_cmd(st)
    res = parse_bmc3(output)
    return res

def bmc3x(ofname, sd, t):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -axvr -S {1:5d} -T {2:5d} -v -L stdout; print_stats\"".format(ofname, sd, t)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    out, output =  run_cmd(st)
    res = parse_bmc3(output)
    return res
