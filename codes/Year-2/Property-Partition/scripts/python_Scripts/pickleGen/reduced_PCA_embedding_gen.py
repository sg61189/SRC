#!/usr/bin/env python3
"""DeepGate2 row aggregate of embedding tensor.

Produces a 128 sized vector with float values by taking an
aggregate on rows of a g x 128 tensor obtained from DeepGate2,
where g is the number of gates in the input circuit.

"""

import argparse
import pickle
from pathlib import Path

import deepgate
import torch
from torch_pca import PCA

def produce_embedding(circuit_file: Path) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = deepgate.Model()
    model.load_pretrained()
    model = model.to(device)

    parser = deepgate.AigParser()
    graph = parser.read_aiger(circuit_file)
    graph = graph.to(device)

    #hs, _ = model(graph)
    hs, hf = model(graph)
    
    
    #print(f"Size of struct tensor {hs.size()}")
    #print(f"Size of funct tensor {hf.size()}")
    #---Transfor the tensor 

    hs_transpose=torch.transpose(hs,0,1)
    hf_transpose=torch.transpose(hf,0,1)

    #print ("After transpose")
    #print(f"Size of struct tensor {hs_transpose.size()}")
    #print(f"Size of funct tensor {hf_transpose.size()}")

    pca_model=PCA(n_components=0.95)

    fitted_struct_tensor=pca_model.fit_transform(hs_transpose)
    fitted_funct_tensor=pca_model.fit_transform(hf_transpose)

    fitted_struct_tensor=torch.transpose(fitted_struct_tensor,0,1)
    fitted_funct_tensor=torch.transpose(fitted_funct_tensor,0,1)
    #return hs_transpose,hf_transpose
    #print ("After transpose-->PCA-->Transpose")
    #print(f"Size of struct tensor {fitted_struct_tensor.size()}")
    #print(f"Size of funct tensor {fitted_funct_tensor.size()}")
    return fitted_struct_tensor,fitted_funct_tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rowavg_embedding.py", description="DeepGate2 embedding tensor using PCA."
    )
    parser.add_argument(
        "-c",
        "--circuit-file",
        help="Path to AIG circuit file.",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "-o1",
        "--outputhspklfile",
        help="Path to pickle file to save 128 sized row-avg'd embedding vector.",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "-o2",
        "--outputhfpklfile",
        help="Path to pickle file to save 128 sized row-avg'd embedding vector.",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    #hs_rowavg = produce_embedding(args.circuit_file)
    hs, hf = produce_embedding(args.circuit_file)

    #with open(args.output_pkl_file, "wb") as pkl_file:
    #    pickle.dump(hs_rowavg, pkl_file)

    with open(args.outputhspklfile, "wb") as hspklFile:
        pickle.dump(hs, hspklFile)

    with open(args.outputhfpklfile, "wb") as hfpklFile:
        pickle.dump(hf, hfpklFile)


if __name__ == "__main__":
    main()
