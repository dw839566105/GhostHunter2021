.PHONY: all

#srcL:=$(shell seq 0 1 3999)
srcL:=$(shell seq 0 1 3999)
dstL:=$(srcL:%=result/Recon_%.h5)

all: $(dstL)
	
result/Recon_%.h5:
	mkdir -p result
	python3 Recon_final.py $* $@ > logs/$*.log 2>&1 || echo exit with non-zero
