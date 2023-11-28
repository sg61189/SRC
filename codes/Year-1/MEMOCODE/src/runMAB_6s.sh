#!/bin/bash
file='6s_seq_test.txt'
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
python3 MAB_BMC_new.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_mabmc_${fn}_D2.txt

#python3 MAB_BMC_frame.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/log_${fn}_D2_frame.txt

#if [["$i" -eq 4]]; then
#fi
#break
i=$((i+1))
done < ${file}
