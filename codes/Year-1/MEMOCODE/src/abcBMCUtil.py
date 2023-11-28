
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple, OrderedDict


DEBUG = True
DEBUG = False

PATH = "../"
abc_result =  namedtuple('abc_result', ['frame', 'var', 'cla', 'conf', 'mem' ,'to', 'asrt', 'tt', 'ld']) 

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
    # print('\t', command) 
    print('\t', output)

    sys.stdout.flush()
    return out, output

def simplify(fname, ofname):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; write {1}\"".format(fname, ofname)
    st = ' '.join([cmdName, "-c", command])
    out, output =  run_cmd(st)


def parse_bmc2(output, sd,t=0):
    ar_tab = OrderedDict()
    sm = None
    xx = r'[ \t]+([\d]+)[ \t]+[:][ \t]+F[ \t]+=[ \t]*([\d]+).[ \t]+O[ \t]+=[ \t]*([\d]+).[ \t]+And[ \t]+=[ \t]*([\d]+).[ \t]+Var[ \t]+=[ \t]*([\d]+).[ \t]+Conf[ \t]+=[ \t]*([\d]+).[ \t]+.*([\d]+)[ \t]+MB[ \t]+([\d]+[.][\d]+)[ \t]+sec'
    m = re.finditer(xx, output, re.M|re.I)
    if DEBUG:
        print(m)
    xx1 = r'No[ \t]+output[ \t]+failed[ \t]+in[ \t]+([\d]+)[ \t]+frames[.]*'
    m2 = re.finditer(xx1, output, re.M|re.I)
    frame_count = -1
    for m21 in m2:
        if DEBUG:
            print(m21.group(1)) 
        frame_count = int(m21.group(1))

    #Output 0 of miter "../benchmark/HWMCC15/6s20_n" was asserted in frame 9
    xx2 = r'Output.+was[ \t]+asserted[ \t]+in[ \t]+frame[ \t]+([\d]+).[.]*'
    m3 = re.finditer(xx2, output, re.M|re.I)
    asrt = -1
    for m31 in m3:
        if DEBUG:
            print(m31.group(1)) 
        asrt = int(m31.group(1))
    pretm = 0
    tt1 = 0
    for m1 in m:
        sm1 = int(m1.group(2)), int(m1.group(5)), int(m1.group(4)), int(m1.group(6)), int(m1.group(7)), float(m1.group(8))
        tt1 = sm1[5] 
        if DEBUG:
            print(sm1, m1.group(1), m21.group(1), asrt)   
        if sm1[3] > 0 and ( asrt > 0 and sm1[0] <= asrt) or (frame_count > 0 and sm1[0] <= frame_count+1 and frame_count >= sd): 
             
            tt = sm1[5] #if t == 0  else t
            to = max(0,sm1[5] - pretm)
            ld = sm1[0]
            if frame_count > 0:
                ld = frame_count
            # if asrt > 0:
            #     ld = asrt
            sm = abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[4], to=to, asrt=asrt, tt = tt, ld=ld) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
            if DEBUG:
                print(sm)
            ar_tab.update({sm.frame:sm})  
        pretm = sm1[5]

    if frame_count > 0:
        key = frame_count + 1  
    else:
        key = sm.frame if (sm and sm is not None) else 0

    print('frame_count', frame_count, key)
    # remove last one as tt > t
    sm = None
    if len(ar_tab.keys()) > 0:
        key = sorted(ar_tab.keys(), reverse = True)[0]
        sm_res = ar_tab[key] 
    else:
        sm_res = sm
    print('sm_res', sm_res)
    res =  asrt, sm_res, ar_tab, tt1
    return res

def parse_bmc3(output, t=0, scale = 1):
    ar_tab = OrderedDict()
    once = True
    sm = None
    xx = r'[ \t]*([\d]+)[ \t]+[+][ \t]+[:][ \t]+Var[ \t]+=[ \t]*([\d]+).[ \t]+Cla[ \t]+=[ \t]*([\d]+).[ \t]+Conf[ \t]+=[ \t]*([\d]+).[ \t]+Learn[ \t]+=[ \t]*([\d]+).[ \t]+.*([\d]+)[ \t]+MB[ \t]*([\d]+)[ \t]+MB[ \t]+([\d]+[.][\d]+)[ \t]+sec'
    m = re.finditer(xx, output, re.M|re.I)
    if DEBUG:
        print(m)

    xx1 = r'No[ \t]+output[ \t]+asserted[ \t]+in[ \t]+([\d]+)[ \t]+frames[.]*'
    m2 = re.finditer(xx1, output, re.M|re.I)
    frame_count = -1
    for m21 in m2:
        if DEBUG:
            print(m21.group(1)) 
        frame_count = int(m21.group(1))
    print(frame_count)
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
    xx3 = r'All[ \t]+([\d]+)[ \t]+outputs[ \t]+are[ \t]+found[ \t]+to[ \t]+be[ \t]+SAT[ \t]+after[ \t]+([\d]+)[ \t]+frames.[.]*'
    asrt_del = -1
    m4 = re.finditer(xx3, output, re.M|re.I)
    for m31 in m4:
        if DEBUG:
            print(m31.group(2)) 
        asrt_del = int(m31.group(2))
    pretm = 0
    tt1 = 0
    for m1 in m:
        sm1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3)), int(m1.group(4)), int(m1.group(5)), int(m1.group(6)), int(m1.group(7)), float(m1.group(8))
        #tm = float(m1.group(7))
        tt1 = sm1[7]*scale
        if DEBUG:
            print(sm1)   
        if sm1[2] > 0 or (frame_count > 0): # and sm1[0] <= frame_count):   
            tt = sm1[7]*scale #if t == 0  else t
            to = max(0, sm1[7] - pretm)

            ld = sm1[0]
            # if frame_count > 0:
            #     ld = frame_count
            # if asrt > 0:
            #     ld = asrt
            sm =  abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = max(sm1[5], sm1[6]), to=to, asrt=asrt, tt=tt, ld=ld) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
            if DEBUG or once:
                print(sm)
                once = False
            if sm.frame in ar_tab:
                sm2 = ar_tab[sm.frame]
                if sm2.to < sm.to:
                    ar_tab.update({sm.frame:sm}) 
            else:
                ar_tab.update({sm.frame:sm}) 
            pretm = ar_tab[sm.frame].tt
         
    if len(ar_tab.keys()) > 0:
        key = sorted(ar_tab.keys(), reverse = True)[0]
        sm1 = ar_tab[key] 
        if asrt_del > 0:
            asrt = sm1.frame
        sm_res =  abc_result(frame=sm1.frame, var=sm1.var, cla=sm1.cla, conf = sm1.conf, mem = sm1.mem, to=sm1.to, asrt=asrt, tt= sm1.tt, ld=sm1.ld)
    else:
        sm_res = sm
    res =  asrt, sm_res, ar_tab, tt1
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

        ld = sm1[1]
        # if frame_count > 0:
        #     ld = frame_count
        # if asrt > 0:
        #     ld = asrt
        sm =  abc_result(frame=sm1[1], conf=0, var=0, cla=0, mem = -1, to=sm1[0], asrt=asrt, tt = t, ld = ld) #, io=(sm2[0], sm2[1]), lat=sm2[2], ag = sm2[3], lev = sm2[4])
        ar_tab.update({sm.frame:sm})  
     
    if len(ar_tab.keys()) > 0:
        key = sorted(ar_tab.keys(), reverse = True)[0]
        sm_res = ar_tab[key] 
    else:
        sm_res = sm
    res =  asrt, sm_res, ar_tab
    return res

def bmc2(ofname, sd, t=0, f=0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc2 -S {1:5d} -T {2:5d} -F {3} -v -L stdout; print_stats\"".format(ofname, sd, t, f)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    print(st)
    out, output =  run_cmd(st)
    res = parse_bmc2(output,sd, t)
    return res

def bmc3(ofname, sd, t=0, f=0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -S {1:5d} -T {2:5d} -F {3} -v -L stdout; print_stats\"".format(ofname, sd, t, f)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]
    out, output =  run_cmd(st)
    res = parse_bmc3(output,t)
    return res

def bmc3rs(ofname, sd, t=0, f=0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -s -S {1:5d} -T {2:5d} -F {3} -v -L stdout; print_stats\"".format(ofname, sd, t, f)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]    
    print(st)
    out, output =  run_cmd(st)
    res = parse_bmc3(output,t)
    return res
   
def bmc3r(ofname, sd, t=0, f=0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -r -S {1:5d} -T {2:5d} -F {3} -v -L stdout; print_stats\"".format(ofname, sd, t, f)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]    
    print(st)
    out, output =  run_cmd(st)
    res = parse_bmc3(output,t)
    return res

def bmc3j(ofname, sd, t=0, f=0, j = 2):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command1 = "\"read {0}; print_stats; &get; bmc3  -S {1:5d} -T {2:5d} -F {4} -J {3} -v -L stdout; print_stats\"".format(ofname, sd, int(t/2), j, f)
    #command = "\"read {0}; print_stats; &get; bmc3 -r -C {1:5d} -D {2:5d} -S {3:5d} -T {4:5d} -v -L stdout; print_stats\"".format(ofname, c, d, sd, t,)
    st1 = ' '.join([cmdName, "-c", command1]) #, "--boound", "20"]    
    print(st1)
    out, output1 =  run_cmd(st1)
    command2 = "\"read {0}; print_stats; &get; bmc3  -S {1:5d} -T {2:5d} -F {4} -J {3} -v -L stdout; print_stats\"".format(ofname,  sd+1, int(t/2), j, f)
    st2 = ' '.join([cmdName, "-c", command2]) #, "--boound", "20"]    
    print(st2)
    out, output2 =  run_cmd(st2)
    res = parse_bmc3('\n'.join([output1,output2]),t, 2)
    # asrt, sm_res, ar_tab = res
    # print('parse result bmc3j')
    # for ky in ar_tab.keys():
    #     print('{0},{1}'.format(ky,ar_tab[ky].to))
    #sm =  abc_result(frame=sm1[0], var=sm1[1], cla=sm1[2], conf = sm1[3], mem = sm1[5],to=sm1[6]-pretm, asrt=asrt)
    return res

def bmc3rg(ofname, sd, t=0, f=0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -g -S {1:5d} -T {2:5d} -F {3} -v -L stdout; print_stats\"".format(ofname, sd, t,f)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]    
    print(st)
    out, output =  run_cmd(st)
    res = parse_bmc3(output,t)
    return res

def bmc3ru(ofname, sd, t=0, f=0):
    pname = os.path.join(PATH, "ABC")
    cmdName = os.path.join(pname, "abc")
    command = "\"read {0}; print_stats; &get; bmc3 -u -S {1:5d} -T {2:5d} -F {3} -v -L stdout; print_stats\"".format(ofname, sd, t, f)
    st = ' '.join([cmdName, "-c", command]) #, "--boound", "20"]    
    print(st)
    out, output =  run_cmd(st)
    res = parse_bmc3(output,t)
    return res

