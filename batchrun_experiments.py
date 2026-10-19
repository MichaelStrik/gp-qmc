from experiments import *
from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel

import os
import json
import time

# manually append jobs here
jobs = []
dirname = os.path.dirname(__file__)
folder = os.path.join(dirname, './exp_data/')
# # artificial functions
# jobs.append(('run_experiment', folder+'artificial_experiment_anchored.json'))
# jobs.append(('run_experiment', folder+'artificial_experiment_mc.json'))
# jobs.append(('run_experiment', folder+'artificial_experiment_unanchored.json'))
# # PDE FEM basis
# jobs.append(('run_feminterpolation_experiment', folder+'feminterpolation_experiment_arcsine.json'))
# jobs.append(('run_feminterpolation_experiment', folder+'feminterpolation_experiment_lognormal.json'))
# jobs.append(('run_feminterpolation_experiment', folder+'feminterpolation_experiment_uniform.json'))

# # PDE quantity of interest
# jobs.append(('run_bvp_experiment', folder+'bvp_experiment_q15_r15.json'))
# jobs.append(('run_bvp_experiment', folder+'bvp_experiment_q15_r22.json'))
# jobs.append(('run_bvp_experiment', folder+'bvp_experiment_q15_r44.json'))
# jobs.append(('run_bvp_experiment', folder+'bvp_experiment_q22_r15.json'))
# jobs.append(('run_bvp_experiment', folder+'bvp_experiment_q22_r22.json'))
# jobs.append(('run_bvp_experiment', folder+'bvp_experiment_q22_r44.json'))
# # multiplication theorem
# jobs.append(('run_multiplication_experiment', folder+'multiplication_experiment_unanchored_lattice.json'))
# jobs.append(('run_multiplication_experiment', folder+'multiplication_experiment_unanchored_mc.json'))


# run jobs
for (exp_fun, exp_path) in jobs:
    # load the experiment
    with open(exp_path, 'r') as file:
        experiment_object = json.load(file)
    experiment_function = locals().get(exp_fun)

    print(f"Started {experiment_object['name']}")

    # replace kernel name by corresponding python class
    if type(experiment_object['kernel_class']) is str:
        kernel_dict = {'UnanchoredSobolevKernel':UnanchoredSobolevKernel, 'AnchoredSobolevKernel':AnchoredSobolevKernel}
        kernel_name = experiment_object['kernel_class']
        experiment_object['kernel_class'] = kernel_dict[kernel_name]
    
    # run the experiment, report and save
    t1 = time.time()
    experiment_object = experiment_function(experiment_object)
    duration = time.time() - t1
    print(f"Successfully finished {experiment_object['name']} with a runtime of {duration//(60*60*24):.0f}d {duration//(60*60)%24:2.0f}h {duration//(60)%60:2.0f}m {duration%60:2.0f}s.")
    write_experiment_data(experiment_object)