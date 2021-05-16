path:=/mnt/stage/douwei/GhostHunter
input:=$(shell seq -f '%02g' 0 1 20)
output_PE:=$(input:%=$(path)/basis_%_PE.h5)
output_Time:=$(input:%=$(path)/basis_%_Time.h5)
output_total:=$(input:%=$(path)/basis_%.h5)

all:$(output_PE) $(output_Time) $(output_total)

$(path)/basis_%_PE.h5:
	mkdir -p $(path)
	python3 basis/main_PE.py $* $@

$(path)/basis_%_Time.h5:
	mkdir -p $(path)
	python3 basis/main_time.py $* $@

$(path)/basis_%.h5:
	mkdir -p $(path)
	python3 basis/main.py $* $@

.DELETE_ON_ERROR:

.SECONDARY:
