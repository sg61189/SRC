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
from bmc_gnn.unfold_circuit import unfold_circuit
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

def process_circuits(end_time: float, args: any, FLAGS: str, CURRENT_FRAME: int, modified_time: int):
    time_stamp = time.time()
    engine = None
    input_circuit_name = os.path.basename(args.input_circuit).split(".")[0]

    # Unfold input circuit
    print(f'Unfolding at depth: {args.UNFOLD_FRAME}\n')
    input_circuit_unfolded = unfold_circuit(
        args.input_circuit, args.UNFOLD_FRAME, args.unfold_path
    )
    # Find most similar circuit and extract frame times
    best_friend = most_similar_circuit(
        input_circuit_unfolded, args.UNFOLD_FRAME, args
    ) 
    engine_list = extract_frame_time(args.UNFOLD_FRAME, args.csv_path, best_friend)
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
    if engine is None:
        modified_time = int(end_time - time.time())
        FLAGS = f"-S {args.UNFOLD_FRAME} -T {modified_time} -F 0 -v"
        print(
            f"No engine found, hence running bmc3g for the remaining time: {modified_time} secs\n"
        )

        depth_reached = run_engine("bmc3g", args.input_circuit, FLAGS 
        )
        print(f'\n{bmc_data_last_depth}\n')
    print(
        f"DEPTH ({args.UNFOLD_FRAME}): DeepGate2 execution + BMC engine selection: {round(time.time() - time_stamp, 2)} secs\n"
        )
    print(
        f"Outcome at DEPTH ({args.UNFOLD_FRAME}): Most similar circuit: {best_friend}.aig, Best BMC engine for {os.path.basename(args.input_circuit)} at Depth {args.UNFOLD_FRAME}: {engine}\n"
    )
    depth_reached = run_engine(engine, args.input_circuit, FLAGS) ###
    print(f'{depth_reached}\n') ###
    if depth_reached is None:
        modified_time = int(end_time - time.time())
        FLAGS = f"-S {args.UNFOLD_FRAME} -T {modified_time} -F 0 -v"
        print(
             f"Since depth_reached is None, running bmc3g for the remaining time: {modified_time} secs\n"
         )
        depth_reached = run_engine("bmc3g", args.input_circuit, FLAGS)
        print(f'\n{bmc_data_last_depth}\n')

    if "+" in depth_reached:
        args.UNFOLD_FRAME = extract_data(depth_reached)[0]
    print(
        f"Running {engine} on {os.path.basename(args.input_circuit)} for {modified_time} seconds, Depth reached: {args.UNFOLD_FRAME}\n"
    )

    if CURRENT_FRAME == args.UNFOLD_FRAME:
        return args.UNFOLD_FRAME, -1, True, engine, depth_reached

    return args.UNFOLD_FRAME, args.UNFOLD_FRAME, True, engine, depth_reached

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
    CURRENT_FRAME = 0
    luby_i = 1
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
                    args.UNFOLD_FRAME, START_FRAME, continue_loop, engine, depth_reached = (
                    process_circuits(end_time, args, FLAGS, CURRENT_FRAME, modified_time))
                    args.UNFOLD_FRAME += 1
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
                    if time_wasted <= args.p*modified_time:
                        print(f'\nTime wasted {time_wasted} < {args.T*args.p}')
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
                            print(f"Running {engine} on {args.input_circuit.split('/')[-1]} for {modified_time} seconds, Depth reached: {args.UNFOLD_FRAME}\n" )
                        if "+" in depth_reached: ###
                            args.UNFOLD_FRAME = extract_data(depth_reached)[0]
                        START_FRAME = args.UNFOLD_FRAME
                        if CURRENT_FRAME == START_FRAME:
                            START_FRAME = -1
                            args.UNFOLD_FRAME +=1
                        else:
                            args.UNFOLD_FRAME +=1
                            START_FRAME +=1
                        
                        first_similarity_check_done = True
                        continue
    
                    else:
                        print("Time wastage exceeded permissible limit, computing new similar circuit")
                        luby_i += 1
                        print(f"\nNext time slot = {args.T} * luby({luby_i}) = {args.T} * {luby(luby_i)} = {args.T * luby(luby_i)}")
                        modified_time = args.T  *  luby(luby_i)
                        FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
                        args.UNFOLD_FRAME, START_FRAME, continue_loop, engine, depth_reached = (
                        process_circuits(end_time, args, FLAGS, CURRENT_FRAME, modified_time)
                        )
                        args.UNFOLD_FRAME += 1
                        if START_FRAME != -1:
                            START_FRAME += 1
                        if not continue_loop:
                            break
                        continue
            else:
                print("No progress, computing new similar circuit")
                START_FRAME = args.UNFOLD_FRAME
                luby_i += 1
                print(f"\nNext time slot = {modified_time} * luby({luby_i}) = {args.T} * {luby(luby_i)} = {args.T * luby(luby_i)}")
                modified_time = args.T * luby(luby_i)
                FLAGS = f"-S {START_FRAME} -T {modified_time} -F 0 -v"
                args.UNFOLD_FRAME, START_FRAME, continue_loop, engine, depth_reached = (
                process_circuits(end_time, args, FLAGS, CURRENT_FRAME, modified_time)
                )
                args.UNFOLD_FRAME += 1
                if START_FRAME != -1:
                    START_FRAME += 1
                if not continue_loop:
                    break
                continue

    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        signal.alarm(0)

'''def variable_time_partition_mode(args: any) -> None:
    with open("data/model.pkl", "rb") as pkl_file:
        model_dict = pickle.load(pkl_file)

    feature_cols = ["Var", "Cla", "Conf", "Learn"]
    start_time = time.time()
    end_time = start_time + args.total_time
    first_iteration = True
    START_FRAME = 0
    CURRENT_FRAME = 0
    first_similarity_check_done = False

    print(
        f"\nTotal execution time: {args.total_time} seconds (Start to End of the Framework)\n"
    )
    signal.signal(signal.SIGALRM, terminate_process)
    signal.alarm(args.total_time)

    try:
        while time.time() - start_time <= args.total_time:
            time_stamp = time.time()
            print("=" * 169 + "\n")

            if (CURRENT_FRAME != START_FRAME and START_FRAME != -1) or first_iteration:
                first_iteration = False
                CURRENT_FRAME = START_FRAME
                FLAGS = f"-S {START_FRAME} -T {args.T} -F 0 -v"

                if first_similarity_check_done:
                    depth_data0 = extract_data(depth_reached[-1])
                    depth_data1 = extract_data(depth_reached[-2])
                    time_wasted = round(args.T - float(depth_data0[7]), 2)
                    print(f"Time wasted in previous iteration: {time_wasted} sec")

                    if time_wasted <= (args.p * args.T):
                        depth_data0 = [int(value) for value in depth_data0[1:5]]
                        depth_data1 = [int(value) for value in depth_data1[1:5]]
                        predictor_data0 = pd.DataFrame(
                            np.asarray(depth_data0).reshape(1, -1), columns=feature_cols
                        )
                        predictor_data1 = pd.DataFrame(
                            np.asarray(depth_data1).reshape(1, -1), columns=feature_cols
                        )
                        predicted_time0 = model_dict[engine].predict(predictor_data0)[0]
                        predicted_time1 = model_dict[engine].predict(predictor_data1)[0]
                        del_T = (predicted_time0 - predicted_time1).item()
                        print(f"\ndel_T: {del_T:.2f} sec")
                        if del_T <= 0:
                            args.T *= 2
                        else:
                            del_T = math.ceil(del_T)
                            args.T = (args.T + del_T) / 2
                        print(f"\nModified time: {args.T:.2f} sec")
                        FLAGS = f"-S {START_FRAME} -T {args.T} -F 0 -v"

                    print(f"\nStarting at DEPTH ({START_FRAME}): \n")
                    depth_reached = run_engine(engine, args.input_circuit, FLAGS)
                    bmc_data_last_depth = depth_reached[-1]

                    if depth_reached[-1] is None:
                        print(
                            f"Running {engine} on {args.input_circuit.split('/')[-1]} for {args.T:.2f} seconds, Depth reached: {args.UNFOLD_FRAME}\n"
                        )
                        print(f"{bmc_data_last_depth}\n")
                        START_FRAME = -1
                        first_similarity_check_done = True
                        continue

                    if "+" in depth_reached[-1]:
                        args.UNFOLD_FRAME = int(
                            depth_reached[-1]
                            .split(":")[0]
                            .strip()
                            .split("+")[0]
                            .strip()
                        )
                    else:
                        for part in depth_reached[-1].split("."):
                            if "F =" in part:
                                args.UNFOLD_FRAME = int(part.split("=")[1].strip())

                    START_FRAME = args.UNFOLD_FRAME
                    if CURRENT_FRAME == START_FRAME:
                        START_FRAME = -1

                    print(
                        f"Running {engine} on {args.input_circuit.split('/')[-1]} for {args.T:.2f} seconds, Depth reached: {args.UNFOLD_FRAME}\n"
                    )
                    print(f"{bmc_data_last_depth}\n")
                    first_similarity_check_done = True
                    continue

            else:
                print("No progress, computing new similar circuit\n")
                args.UNFOLD_FRAME, START_FRAME, continue_loop, engine, depth_reached = (
                    process_circuits(end_time, args, FLAGS, CURRENT_FRAME)
                )
                if not continue_loop:
                    break
                continue

            args.UNFOLD_FRAME, START_FRAME, continue_loop, engine, depth_reached = (
                process_circuits(end_time, args, FLAGS, CURRENT_FRAME)
            )
            if not continue_loop:
                break
            first_similarity_check_done = True

    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        signal.alarm(0)'''

def valid_range(value):
    return float(value) if 0 <= float(value) <= 1 else argparse.ArgumentTypeError(f"Value must be between 0 and 1. Provided value: {value}")

def main():
    global args
    initial_parser = argparse.ArgumentParser(
        description="BMC Sequence Script", add_help=False
    )
    group = initial_parser.add_mutually_exclusive_group(required=True)
    #group.add_argument("-v", action="store_true", help="Run in mode --> Variable")
    group.add_argument("-f", action="store_true", help="Run in mode --> Fixed")
    initial_args, remaining_argv = initial_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="BMC Sequence Script")
    parser.add_argument(
        "-input_circuit", type=str, help="Name of the input circuit", required=True
    )
    # parser.add_argument("-known_circuit_path", type=str, help="Path to the known circuits directory", required=True)
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
        "-UNFOLD_FRAME", type=int, help="Initial unfolding frame", required=True
    )

    parser.add_argument(
        "-chosen_circuit_path", type=str, help="chosen_circuit_path_to_.pkl files.", required=True
    )



    args = parser.parse_args(remaining_argv)

    if initial_args.f:
        fixed_time_partition_mode(args)
    #elif initial_args.v:
        #variable_time_partition_mode(args)
    else:
        print("Please specify a mode with -f")


if __name__ == "__main__":
    main()
