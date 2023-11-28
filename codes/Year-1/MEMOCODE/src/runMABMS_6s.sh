#!/bin/bash
file='6s_seq_test.txt'
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
python3 MABMS_BMC.py -i benchmark/HWMCC_15_17/${fn}.aig > logs/logMS_${fn}_D2.txt
i=$((i+1))

#break
done < ${file}
