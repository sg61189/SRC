#!/usr/bin/bash
set -euo pipefail

usage () {
  echo "Return the max depth reached by any bmc engine for a given circuit.

Usage:
$(basename "$0") [-h] -c <circuit_name>

where:
    -h      Print this message and exit.
    -c <aig_file>
    File name for the input circuit (without '.aig')." >&2
}

while getopts ':hc:' option; do
  case "$option" in
    h) usage; exit;;
    c) CIRCUIT="$OPTARG";;
    :) echo "missing argument for -$OPTARG" >&2; usage; exit 1;;
   \?) echo "unknown option: -$OPTARG" >&2; usage; exit 1;;
    *) echo "unimplemented option: -$option" >&2; usage; exit 1;;
  esac
done
shift $((OPTIND - 1))

if [ -z "${CIRCUIT+x}" ]; then
  echo "One or more mandatory options missing." >&2; usage; exit 1
fi

sort <(for i in ../data/bmc_data_csv/bmc*/"${CIRCUIT}.csv"; do tail -n 1 "${i}"; done) | tail -n 1 | cut -f 1 -d,
