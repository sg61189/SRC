

Installation:
- Python packages:
 scipy, numpy, matplotlib, pandas, scikit-learn
- ABC 

Some modifications in ABC output for logs regrading conflict clauses (needs to compile the source codes)
>> cd ABC

>> make all


----------------- Running MAB_BMC with time slices predicted ------------------

>> cd src

>> python3 MABMC_top.py -i ../../benchmark/HWMCC_15_17/6s7.aig > results/log_mabmc_fix_IF_6s7.txt
