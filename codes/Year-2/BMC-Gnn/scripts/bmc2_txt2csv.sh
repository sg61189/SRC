#!/usr/bin/bash
set -euo pipefail

usage () {
  echo "Convert bmc2 output text to CSV.

Usage:
$(basename "$0") [-h] -t <bmc_text>

where:
    -h      Print this message and exit.
    -t <bmc_text>
            BMC output text file." >&2
}

while getopts ':ht:' option; do
  case "$option" in
    h) usage; exit;;
    t) BMC_TEXT="$OPTARG";;
    :) echo "missing argument for -$OPTARG" >&2; usage; exit 1;;
   \?) echo "unknown option: -$OPTARG" >&2; usage; exit 1;;
    *) echo "unimplemented option: -$option" >&2; usage; exit 1;;
  esac
done
shift $((OPTIND - 1))

if [ -z "${BMC_TEXT+x}" ]; then
  echo "One or more mandatory options missing." >&2; usage; exit 1
fi

echo 'F,O,And,Var,Conf,Cla,Learn,Memory,Time'

tail -n +6 < "${BMC_TEXT}" | \
head -n -3 | \
sed 's/.*://' | \
sed 's/[^0-9\ \.]*//g' | \
sed 's/\ $//' | \
sed 's/^\ \+//' | \
sed 's/\([0-9]\)\.\ /\1/g' | \
sed 's/\ \+/,/g'
