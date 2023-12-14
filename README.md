# Quantile-Vector-Autoregression-with-Influencers-and-Communities
This is a solution algorithm for a novel high-dimensional quantile vector autoregression (QVAR) model accompanied by influencers and communities.<br>
**"prepare.py"** contains functions used for preparatory work before the algorithm execution, mainly functions to generate cluster indices(get_index_matrix( )), randomize initial positions(random_basis( )), and compute inter-cluster distances(index_dist( )).<br>
**"alternating.py"** contains frequently used functions during execution, mainly including the kernel function, loss function, and the two most crucial alternating iterative functions: v_new_step( ) and z_new_step( ). **It's worth noting that the selection of the quantile τ is specified within this file.**<br>
**"main.py"** is the primary code for the algorithm, which includes data reading, alternating iterative optimization of the loss function, and output of the model results.<br>
**"multicode.py"** is a multithreaded version of "main.py". Users can utilize this code to accelerate computations using multithreading.<br>
**"stability.py"** is designed to take stability test results as input and produce the corresponding dot plots as output.<br>
**"Realdata"** folder contains the macroeconomic data used in empirical analysis.<br>
**"ecoheapmap"** folder stores the visualization heatmaps of the model fitting results under different quantiles.<br>
**"stability"** folder stores the results of stability analysis for different numbers of clusters.<br>
