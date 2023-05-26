#!/bin/bash
to=3600
i=1
fn=6s13
echo "----------------- Running MAB_bmc_fixed time ${fn} ------------------"
python3 MAB_BMC_fixt_ucb1.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_mabmc_fix_ucb1_IF_${fn}_D3.txt

fn=6s309b034
echo "----------------- Running MAB_bmc_fixed time ${fn} ------------------"
python3 MAB_BMC_fixt_ucb1.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_mabmc_fix_ucb1_IF_${fn}_D3.txt

fn=6s320rb0
echo "----------------- Running MAB_bmc_fixed time ${fn} ------------------"
python3 MAB_BMC_fixt_ucb1.py -i benchmark/HWMCC_15_17/${fn}.aig > logs_IF/log_mabmc_fix_ucb1_IF_${fn}_D3.txt

#break
i=$((i+1))
