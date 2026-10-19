podman run -d \
	--replace \
	--name gp-qmc-batch-experiments \
	-v "$(pwd)":/shared \
	-w /shared \
	fenicsx_custom \
	python3 batchrun_experiments.py
