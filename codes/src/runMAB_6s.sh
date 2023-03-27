#!/bin/bash
file='6snames_seq.txt'
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
python3 MAB_BMC.py -i benchmark/HWMCC15/${fn}.aig > logs/log_${fn}_D2.txt
i=$((i+1))


done < ${file}
