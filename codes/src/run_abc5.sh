#!/bin/bash
file='6s_seq_testn5.txt'
to=3600
i=1
while read line; do
#Reading each line.
echo "----------------- ${i} Running file ${line} ------------------"
fn=${line}
echo "-----------------  bmc3g ------------------"
../ABC/abc -c "read ../benchmark/HWMCC_15_17/${fn}.aig; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; bmc3 -g -S 0 -T ${to} -v -L stdout; print_stats;" > logs_IF/log_abc_bmc3g_${fn}.txt

echo "-----------------  bmc2 ------------------"
../ABC/abc -c "read ../benchmark/HWMCC_15_17/${fn}.aig; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; bmc2 -S 0 -T ${to} -v -L stdout; print_stats;" > logs_IF/log_abc_bmc2_${fn}.txt

echo "-----------------  bmc3 ------------------"
../ABC/abc -c "read ../benchmark/HWMCC_15_17/${fn}.aig; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; bmc3 -S 0 -T ${to} -v -L stdout; print_stats;" > logs_IF/log_abc_bmc3_${fn}.txt

echo "-----------------  bmc3s ------------------"
../ABC/abc -c "read ../benchmark/HWMCC_15_17/${fn}.aig; print_stats; &get; &dc2;&put;dretime;&get;&lcorr;&dc2;&put;dretime;&get;&scorr;&fraig;&dc2;&put;dretime;&put; print_stats; &get; bmc3s -S 0 -T ${to} -v -L stdout; print_stats;" > logs_IF/log_abc_bmc3s_${fn}.txt

#fi
#break
i=$((i+1))
done < ${file}
