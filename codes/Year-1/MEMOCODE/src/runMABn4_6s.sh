#!/bin/bash
file='6s_seq_testn4.txt'
to=3600
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
echo "----------------- Running MAB_bmc_fixed time ------------------"
python3 MAB_BMC_fixt_n.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_mabmc_fix_IF_${fn}_D3.txt
echo "----------------- Running MAB_bmc_new ------------------"
python3 MAB_BMC_new_n.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_mabmc_IF_${fn}_D3.txt

# echo "----------------- Separate abc ------------------"
# python3 run_abc.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_abc_${fn}.txt
#echo "----------------- Running MAB_bmc_frame ------------------"
#python3 MAB_BMC_fo.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_mabmc_fo_${fn}_D3.txt
# echo "----------------- Running Superdeep ------------------"
# (time (timeout ${to} ../super-prove-build/build/super_prove/bin/super_deep.sh -r stdout ../benchmark/HWMCC_15_17/${fn}.aig) 2>&1) > logs/log_deep_${fn}_D2.txt 

#python3 MAB_BMC_frame.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_${fn}_D2_frame.txt

# if [ "${fn}" = "6s366r" ]; then
# echo "----------------- Running Superdeep ------------------"
# (time (timeout ${to} ../super-prove-build/build/super_prove/bin/super_deep.sh -r stdout ../benchmark/HWMCC_15_17/${fn}.aig) 2>&1) > logs_IF/log_deep_${fn}_D2.txt 

# fi

# if [ "${fn}" = "6s376r" ]; then
# echo "----------------- Running Superdeep ------------------"
# (time (timeout ${to} ../super-prove-build/build/super_prove/bin/super_deep.sh -r stdout ../benchmark/HWMCC_15_17/${fn}.aig) 2>&1) > logs_IF/log_deep_${fn}_D2.txt 

fi

i=$((i+1))
done < ${file}
