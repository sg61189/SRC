

Installation:
- Python packages:
 scipy, numpy, matplotlib, pandas, scikit-learn
- ABC 


----------------- Running MAB_bmc_fixed time slices 60, 120, 240, ------------------

>> cd src


>> python3 MAB_BMC_fixt_n.py -i ../../benchmark/HWMCC_15_17/6s7.aig > results/log_mabmc_fix_IF_6s7_D3.txt

----------------- Running MAB_bmc_new time slices 60, 60, 120, 120, 120, 240 ------------------

>> cd src


>> python3 MAB_BMC_new_n.py -i ../../benchmark/HWMCC_15_17/6s7.aig > results/log_mabmc_IF_6s7_D3.txt

----------------- Running single BMC engines from ABC ------------------

>> cd src


>> python3 run_abc.py -i ../../benchmark/HWMCC_15_17/6s7.aig > results/log_abc_6s7.txt

