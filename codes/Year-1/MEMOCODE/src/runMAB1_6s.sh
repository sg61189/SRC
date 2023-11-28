#!/bin/bash
file='6s_seq_test.txt'
to=3600
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
python3 MABMC_fixt_to.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/new_log_mabmc_ft_${fn}.txt
# mprof run  python3 MAB_BMC_fixt_n.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_mabmc_fixt_n_${fn}_D2.txt
# mprof plot -o plots/mprof_MAB_BMC_fixt_n_${fn}.png
# mprof run python3 MAB_BMC_new_n.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_mabmc_new_n_${fn}_D2.txt
# mprof plot -o plots/mprof_MAB_BMC_new_n_${fn}.png
# #python3 MAB_BMC_fo.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_mabmc_fo_${fn}_D2.txt
# (time (timeout ${to} ../super-prove-build/build2/super_prove/bin/super_deep.sh -r stdout ../benchmark/HWMCC_15_17/${fn}.aig) 2>&1) > logs/log_deep_${fn}_D2.txt 
# mprof plot -o plots/mprof_super_deep_${fn}.png
#python3 MAB_BMC_frame.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_${fn}_D2_frame.txt

# if [["$i" -eq 2]]; then
# 	break
# fi
i=$((i+1))
done < ${file}
