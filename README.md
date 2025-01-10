# Repository for the Term Project of Machine Learning, Fall 2024, PKU

## Problem
For dimensions $n$, design Lattice such that the normalized quantization error is small.

## Run
You can run the following command to search for the best Lattice for a given dimension. Please refer to the `examples` directory for the configuration file. Feel free to modify the configuration file to suit your needs.
```bash
python sgd.py --config examples/medium.yml 12 # dimension
```

You can run the following command to compute Normalized Second Moment (NSM) of a given lattice. We set the number of samples to $10^9$ defaultly.
```bash
python check_nsm.py --basis "<dir_to_your_lattice>" --num_samples <the_number_of_samples>
```

And we use `draw_figs.py` to draw the figures of the comparison of the NSM of different lattices. You can refer to `draw_figs.py` for more details.

## Results
We put all the results in the `data` directory, including the generation matrix of the lattices with settings:
- dimension $[1,25]$, batch size = $1$, default scheduler, i.e. the reproduction of the results in [Optimization and Identification of Lattice Quantizers](https://arxiv.org/abs/2401.01799) 
- dimension $[1,25]$, batch size = $8$, default scheduler. 
And the comparison of dimension $12$ lattice convergence between the reproduced algorithm with batch = 1 and the batch algorithm with batch = 8. We obtain the results in the medium mode with the default scheduler ($v = 10$) and timesteps $10^6$.

