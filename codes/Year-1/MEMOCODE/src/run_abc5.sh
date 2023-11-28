#!/bin/bash
file='6s_seq_testn5.txt'
to=3600
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}

echo "----------------- Separate abc ------------------"
python3 run_abc.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_abc_${fn}.txt
#fi
#break
i=$((i+1))
done < ${file}
