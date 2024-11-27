#!/bin/bash

files=("6s145.aig" "6s148.aig" "6s160.aig" "6s188.aig" "6s191.aig" "6s24.aig" "6s339rb19.aig" "6s343b31.aig" "6s351rb02.aig" "6s366r.aig" "6s39.aig" "6s44.aig" "intel013.aig" "intel025.aig" "bobsmcodic.aig" "6s341r.aig")
#files=("6s145.aig" "6s188.aig" "6s339rb19.aig" "6s351rb02.aig" "6s44.aig")
input_dir="/home/src2024/bmc-gnn2/data/mab_bmc_unsat_circuits_reported"
csv_path="/home/src2024/bmc-gnn2/data/train_data_csv"
chosen_circuit_dir="/home/src2024/bmc-gnn2/data/chosen_circuits_non-inductive"

for file in "${files[@]}"; do
    
    python3 -u ~/bmc-gnn2/main_luby_noninductive.py -f -input_circuit "${input_dir}/${file}" -csv_path "${csv_path}" -unfold_path /tmp -total_time 3600 -p 0.2 -chosen_circuit_path "${chosen_circuit_dir}" -T 60 -UNFOLD_FRAME 1 | tee -a /home/src2024/bmc-gnn2/results/luby/non-inductive/17_luby_non-inductive_0.2.txt

done
