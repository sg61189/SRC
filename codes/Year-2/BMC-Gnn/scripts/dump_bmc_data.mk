#!/usr/bin/make -f

TIMELIMIT ?= 3600
FLAGS := -S 0 -T ${TIMELIMIT} -F 0 -v

ABC ?= abc
PREFIX ?= ../data/bmc_data_circuits

SRCS := $(wildcard ${PREFIX}/*.aig)
OBJS := $(notdir ${SRCS:.aig=.txt})

all: bmc2 bmc3 bmc3r bmc3s bmc3g bmc3u bmc3j

bmc2: $(foreach obj,$(OBJS),${PREFIX}/bmc2/$(obj))
bmc3: $(foreach obj,$(OBJS),${PREFIX}/bmc3/$(obj))
bmc3r: $(foreach obj,$(OBJS),${PREFIX}/bmc3r/$(obj))
bmc3s: $(foreach obj,$(OBJS),${PREFIX}/bmc3s/$(obj))
bmc3g: $(foreach obj,$(OBJS),${PREFIX}/bmc3g/$(obj))
bmc3u: $(foreach obj,$(OBJS),${PREFIX}/bmc3u/$(obj))
bmc3j: $(foreach obj,$(OBJS),${PREFIX}/bmc3j/$(obj))

${PREFIX}/bmc2/%.txt: ${PREFIX}/%.aig | ${PREFIX}/bmc2/
	${ABC} -c "read $<; print_stats; &get; bmc2 ${FLAGS}; print_stats" > "$@"
${PREFIX}/bmc3/%.txt: ${PREFIX}/%.aig | ${PREFIX}/bmc3/
	${ABC} -c "read $<; print_stats; &get; bmc3 ${FLAGS}; print_stats" > "$@"
${PREFIX}/bmc3r/%.txt: ${PREFIX}/%.aig| ${PREFIX}/bmc3r/
	${ABC} -c "read $<; print_stats; &get; bmc3 -r ${FLAGS}; print_stats" > "$@"
${PREFIX}/bmc3s/%.txt: ${PREFIX}/%.aig | ${PREFIX}/bmc3s/
	${ABC} -c "read $<; print_stats; &get; bmc3 -s ${FLAGS}; print_stats" > "$@"
${PREFIX}/bmc3g/%.txt: ${PREFIX}/%.aig | ${PREFIX}/bmc3g/
	${ABC} -c "read $<; print_stats; &get; bmc3 -g ${FLAGS}; print_stats" > "$@"
${PREFIX}/bmc3u/%.txt: ${PREFIX}/%.aig | ${PREFIX}/bmc3u/
	${ABC} -c "read $<; print_stats; &get; bmc3 -u ${FLAGS}; print_stats" > "$@"
${PREFIX}/bmc3j/%.txt: ${PREFIX}/%.aig | ${PREFIX}/bmc3j/
	${ABC} -c "read $<; print_stats; &get; bmc3 ${FLAGS} -J 2; print_stats" > "$@"

${PREFIX}/%/:
	mkdir -p "$@"

clean:
	-rm -r ${PREFIX}/bmc2/
	-rm -r ${PREFIX}/bmc3/
	-rm -r ${PREFIX}/bmc3r/
	-rm -r ${PREFIX}/bmc3s/
	-rm -r ${PREFIX}/bmc3g/
	-rm -r ${PREFIX}/bmc3u/
	-rm -r ${PREFIX}/bmc3j/

.PHONY: all clean
