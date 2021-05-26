.PHONY: all

srcL:=$(shell seq 0 1 4000)
dstL:=$(srcL:%=result/Recon_%.h5)

all: $(dstL)
	
result/Recon_%.h5:
	python3 Recon_final.py $* $@
