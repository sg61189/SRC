#!/usr/bin/make -f

ABC ?= abc
CIRCUIT ?= ../data/chosen_circuits/6s7.aig
MAX_DEPTH ?= 10
PREFIX ?= /tmp

PICKLES := $(foreach n, $(shell seq -w 1 1), ${PREFIX}/$(notdir $(basename ${CIRCUIT}))_$(n).pkl)

all: ${PICKLES}
	
%.pkl: %.aig ${PREFIX}/
	python3 ./rowavg_embedding.py -c "$<" -o "$@" && rm "$<"

%.aig: ${PREFIX}/
	./unroll_circuit.sh -c "${CIRCUIT}" -u "$@"

${PREFIX}/:
	mkdir -p "${PREFIX}"

clean:
	-rm -f ${PICKLES}

.INTERMEDIATE: %.aig
.PHONY: all clean
