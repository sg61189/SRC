#!/home/prateek/deepgate2-venv/bin/python3
import argparse
import subprocess
import os
import random
import re
import math
import signal
import time
import pandas as pd
import numpy as np
import pickle

from bmc_gnn.extract_frame_time import extract_frame_time
from bmc_gnn.most_similar_circuit import most_similar_circuit
from bmc_gnn.unfold_circuit_inductive import unfold_circuit
from bmc_gnn.luby import luby

bmc_data_last_depth: str = ""


def run_engine(selected_engine: str, circuit: str, FLAGS: str) -> list[str]:
    global bmc_data_last_depth
    output = []
    max_depth = None

    if selected_engine == "bmc3j":
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; bmc3 {FLAGS} -J 2; print_stats",
        ]
    elif selected_engine == "bmc3":
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; bmc3 {FLAGS}; print_stats",
        ]
    elif selected_engine == "bmc3g":
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; bmc3 -g {FLAGS}; print_stats",
        ]
    elif selected_engine == "bmc3s":
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; bmc3 -s {FLAGS}; print_stats",
        ]
    elif selected_engine == "bmc3u":
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; bmc3 -u {FLAGS}; print_stats",
        ]
    else:
        raise ValueError(f"Unknown engine: {selected_engine}")

    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    while True:
        line = process.stdout.readline()
        if not line:
            break

        line = line.strip()

        expected_start = f'Output 0 of miter "{circuit.split(".")[0]}" was asserted'
        if line.startswith(expected_start):
            print(f"Found: {line}", flush=True)
            os._exit(1)

        if any(line.startswith(x) for x in ["Reached", "Runtime", "No output"]):
            continue

        if re.match(r"^\d+ \+ :", line):
            bmc_data_last_depth = line
            output.append(line)
            max_depth = line
    return max_depth

def process_circuits(START_FRAME: int, end_time: float, args: any, FLAGS: str, CURRENT_FRAME: int, modified_time: int, k: int):
    time_stamp = time.time()
    engine = None
    input_circuit_name = os.path.basename(args.input_circuit).split(".")[0]

    # Unfold input circuit
    print(f'Unfolding at depth: {k+1}\n')
    input_circuit_unfolded = unfold_circuit(
        args.input_circuit, k+1, args.unfold_path
    )
    print(f"DEPTH ({k+1}): DeepGate2 execution + BMC engine selection: {round(time.time() - time_stamp, 2)} secs\n"
        )
    print(f"{FLAGS}")
    # Find most similar circuit and extract frame times
    best_friend = most_similar_circuit(
            input_circuit_unfolded, k+1, args
    ) 
    if START_FRAME == 0:
        engine_list = extract_frame_time(k+1, args.csv_path, best_friend)
    else:
        print(f"Finding friend at {START_FRAME}")
        engine_list = extract_frame_time(START_FRAME, args.csv_path, best_friend)
    # Select engine with minimum time
    if engine_list:
        min_time = min(engine[2][0] for engine in engine_list)
        selected_engines = [engine for engine in engine_list if engine[2][0] == min_time]
        formatted_selected_engines =  '     '.join(f"({i[0]} | {i[1]} | {i[2][0]})" for i in selected_engines)
        print(f"{formatted_selected_engines}\n")
        for e in selected_engines:
            if 'bmc3g' == e[0]:
                engine = e[0]
                break
            else:
                engine = random.choice(selected_engines)[0]
    print(
        f"Outcome at DEPTH ({k+1}): Most similar circuit: {best_friend}.aig, Best BMC engine for {os.path.basename(args.input_circuit)} at Depth {START_FRAME}: {engine}\n"
    )
    if engine is None:
        modified_time = int(end_time - time.time())
        FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
        print(
            f"No engine found, hence running bmc3g for the remaining time: {modified_time} secs\n"
        )

        depth_reached = run_engine("bmc3g", args.input_circuit, FLAGS 
        )
        print(f'\n{bmc_data_last_depth}\n')
        os._exit(1)

    depth_reached = run_engine(engine, args.input_circuit, FLAGS) ###
    k+=1
    print(f'{depth_reached}\n') ###
    if depth_reached is None:
        modified_time = int(end_time - time.time())
        FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
        print(
             f"Since depth_reached is None, running bmc3g for the remaining time: {modified_time} secs\n"
         )
        depth_reached = run_engine("bmc3g", args.input_circuit, FLAGS)
        print(f'\n{bmc_data_last_depth}\n')
        os._exit(1)
    if "+" in depth_reached:
        START_FRAME = extract_data(depth_reached)[0]
    print(
        f"Running {engine} on {os.path.basename(args.input_circuit)} for {modified_time} seconds, Depth reached: {START_FRAME}\n"
    )

    if CURRENT_FRAME == START_FRAME:
        return START_FRAME, -1, True, engine, depth_reached, k

    return START_FRAME, START_FRAME, True, engine, depth_reached, k

def extract_data(data: str):
    predictors = re.sub(r"[^0-9 \.]", "", data)
    predictors = re.sub(r"\s+", " ", predictors)
    predictors = re.sub(r"(\d)\. ", r"\1 ", predictors)
    predictors = predictors.strip()
    predictors = predictors.replace(" ", ",")
    predictors = re.split(",", predictors)
    predictors = [float(num) if "." in num else int(num) for num in predictors]
    return predictors

def terminate_process(signum: int, frame: any) -> None:
    global bmc_data_last_depth
    print("=" * 80 + " TIMEOUT " + "=" * 80)
    input = os.path.basename(args.input_circuit).split('.')[0]
    print(f"\n{bmc_data_last_depth}\n")
    os._exit(1)

def fixed_time_partition_mode(args: any) -> None:
    global bmc_data_last_depth
    start_time = time.time()
    modified_time = args.T
    end_time = start_time + args.total_time
    first_iteration = True
    START_FRAME = 0
    MODIFIED_START_FRAME = 0
    CURRENT_FRAME = 0

    luby_i = 1
    k = 0
    first_similarity_check_done = False

    print(
        f"\nTotal execution time: {args.total_time} seconds (Start to End of the Framework)\n"
    )
    signal.signal(signal.SIGALRM, terminate_process)
    signal.alarm(args.total_time)

    try:
        while time.time() - start_time <= args.total_time:
            print("=" * 169 + "\n")

            if (
                (CURRENT_FRAME != START_FRAME and START_FRAME != -1)
                or first_iteration
                or START_FRAME != -1
            ):
                first_iteration = False
                CURRENT_FRAME = START_FRAME
                FLAGS = f"-S {START_FRAME} -T {args.T} -F 0 -v"
                if first_similarity_check_done is not True:
                    MODIFIED_START_FRAME, START_FRAME, continue_loop, engine, depth_reached, k = (
                    process_circuits(START_FRAME,end_time, args, FLAGS, CURRENT_FRAME, modified_time, k))
                    MODIFIED_START_FRAME += 1
                    if START_FRAME != -1:
                        START_FRAME += 1
                    if not continue_loop:
                        break
                    first_similarity_check_done = True
                    continue


                if first_similarity_check_done:
                    last_depth_data = extract_data(depth_reached) ###
                    time_wasted = round(modified_time - float(last_depth_data[7]), 2)
                    print(f"Time wasted in previous iteration: {time_wasted} sec")
                    if time_wasted < args.p*modified_time:
                        print(f'\nTime wasted {time_wasted} < {args.p*modified_time}')
                        luby_i += 1
                        print(f"\nNext time slot = {args.T} * luby({luby_i}) = {args.T} * {luby(luby_i)} = {args.T * luby(luby_i)}")
                        modified_time = args.T * luby(luby_i)
                        FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
                        print(f"\nStarting at DEPTH ({START_FRAME}): \n")
                        
                        depth_reached = run_engine(engine, args.input_circuit, FLAGS)
                        print(f'{depth_reached}\n') ###

                        if depth_reached is None: ###
                            modified_time = int(end_time - time.time())
                            FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
                            print(
                            f"Since depth_reached is None, running bmc3g for the remaining time: {modified_time} secs\n"
                            )
                            depth_reached = run_engine("bmc3g", args.input_circuit, FLAGS)
                            print(f"{bmc_data_last_depth}\n")

                            break
                        
                        if "+" in depth_reached: ###
                            MODIFIED_START_FRAME = extract_data(depth_reached)[0]
                        START_FRAME = MODIFIED_START_FRAME
                        print(f"Running {engine} on {args.input_circuit.split('/')[-1]} for {modified_time} seconds, Depth reached: {MODIFIED_START_FRAME}\n" )
                        if CURRENT_FRAME == START_FRAME:
                            START_FRAME = -1
                            MODIFIED_START_FRAME +=1
                        else:
                            MODIFIED_START_FRAME +=1
                            START_FRAME +=1
                        
                        first_similarity_check_done = True
                        continue
    
                    else:
                        print("\nTime wastage exceeded permissible limit, computing new similar circuit")
                        
                        luby_i += 1
                        print(f"\nNext time slot = {args.T} * luby({luby_i}) = {args.T} * {luby(luby_i)} = {args.T * luby(luby_i)}")
                        modified_time = args.T * luby(luby_i)

                        FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
                        MODIFIED_START_FRAME, START_FRAME, continue_loop, engine, depth_reached, k = (
                        process_circuits(START_FRAME, end_time, args, FLAGS, CURRENT_FRAME, modified_time, k)
                        )
                        MODIFIED_START_FRAME += 1
                        if START_FRAME != -1:
                            START_FRAME += 1
                        if not continue_loop:
                            break
                        continue
            else:
                last_depth_data = extract_data(depth_reached) ###
                time_wasted = round(modified_time - float(last_depth_data[7]), 2)
                print(f"\nTime wasted in previous iteration: {time_wasted} sec")
                print("\nNo progress, computing new similar circuit")
                START_FRAME = MODIFIED_START_FRAME
                CURRENT_FRAME = MODIFIED_START_FRAME
                luby_i += 1
                print(f"\nNext time slot = {args.T} * luby({luby_i}) = {args.T} * {luby(luby_i)} = {args.T * luby(luby_i)}")
                modified_time = args.T * luby(luby_i)
                FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
                MODIFIED_START_FRAME, START_FRAME, continue_loop, engine, depth_reached, k = (
                process_circuits(START_FRAME,end_time, args, FLAGS, CURRENT_FRAME, modified_time, k)
                )
                MODIFIED_START_FRAME += 1
                if START_FRAME != -1:
                    START_FRAME += 1
                if not continue_loop:
                    break
                continue

    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        signal.alarm(0)

def valid_range(value):
    return float(value) if 0 <= float(value) <= 1 else argparse.ArgumentTypeError(f"Value must be between 0 and 1. Provided value: {value}")

def main():
    global args
    initial_parser = argparse.ArgumentParser(
        description="BMC Sequence Script", add_help=False
    )
    group = initial_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", action="store_true", help="Run in mode --> Fixed")
    initial_args, remaining_argv = initial_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="BMC Sequence Script")
    parser.add_argument(
        "-input_circuit", type=str, help="Name of the input circuit", required=True
    )
    parser.add_argument(
        "-csv_path", type=str, help="Path to the CSV directory", required=True
    )
    parser.add_argument(
        "-unfold_path", type=str, help="Path to the unfold directory", required=True
    )
    parser.add_argument(
        "-T", type=int, help="Time duration for initial iteration", required=True
    )
    parser.add_argument(
        "-p", type=valid_range, help="Maximum allowed time waste permitted.", default=0.4
    )
    parser.add_argument(
        "-total_time",
        type=int,
        help="Total time period of the execution",
        required=True,
    )
    parser.add_argument(
        "-chosen_circuit_path", type=str, help="chosen_circuit_path_to_.pkl files.", required=True
    )

    args = parser.parse_args(remaining_argv)

    if initial_args.f:
        fixed_time_partition_mode(args)

    else:
        print("Please specify a mode with -f")


if __name__ == "__main__":
    main()
