#!/usr/bin/bash
set -euo pipefail

usage () {
  echo "Unroll a circuit (say, 6s7.aig) to produce a new AIG file (say, 6s7_3.aig)

Usage:
$(basename "$0") [-h] -c <orig_aig_file> -u <unrolled_aig_file>

where:
    -h      Print this message and exit.
    -c <orig_aig_file>
            AIG file for the input circuit.
    -u <unrolled_aig_file>
            AIG file to unroll to (must be of form *_<DEPTH>.aig)" >&2
}

while getopts ':hc:u:' option; do
  case "$option" in
    h) usage; exit;;
    c) ORIG_CIRCUIT="$OPTARG";;
    u) UNROLLED_CIRCUIT="$OPTARG";;
    :) echo "missing argument for -$OPTARG" >&2; usage; exit 1;;
   \?) echo "unknown option: -$OPTARG" >&2; usage; exit 1;;
    *) echo "unimplemented option: -$option" >&2; usage; exit 1;;
  esac
done
shift $((OPTIND - 1))

if [ -z "${ORIG_CIRCUIT+x}" ] || [ -z "${UNROLLED_CIRCUIT+x}" ]; then
  echo "One or more mandatory options missing." >&2; usage; exit 1
fi

DEPTH=$(echo "${ORIG_CIRCUIT}" | sed -e 's/\.aig$//; s/^.*_//; s/^0*//')
#abc -c "read ${ORIG_CIRCUIT}; &get; &frames -F ${DEPTH} -s -b; &write ${UNROLLED_CIRCUIT}"
abc -c "read ${ORIG_CIRCUIT}; &get; &frames -F ${DEPTH} -s -b -i; &write ${UNROLLED_CIRCUIT}"
