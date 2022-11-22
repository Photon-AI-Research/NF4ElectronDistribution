# Normalizing Flows for Reconstruction of Electron Bunch Distribution along a FEL Beamline

We apply normalizing flows to reconstrunct distribution of electrons in a bunch while propagating through an FEL beamline. 

### Simulations

Data is generated using [the Advanced Photon Source](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software) elegant software. 
In our work we studied dependencies of electron cloud transformations for differenet emmitance and beta values in x and y directions. 
Example of simulation code is located in ./simulations directory, in order to run simulations to create data used in our analysis, run create_data.sh.
Data will appear in ./simulations/data directory. The APS elegant library has to be installed.

### Models

In our work there are implemented the following models:

* [conditional Invertible Neural Network](https://arxiv.org/abs/1907.02392) using [FrEIA](https://github.com/vislearn/FrEIA) library
* [Masked Autoregressive Flows(MAF)](https://papers.nips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html) using [nflows](https://github.com/bayesiains/nflows) library

Classes of models are described in ./models/model_NF.py and ./models/model_MAF.py correspondingly.
MAF model supports parallel distributed training using [horovod](https://github.com/horovod/horovod) library.

During training one can track evolution of loss and watch customized plots of reconstructions using wandb.ai

### Requirements
    cuda/10.2 \
    python/3.6.5 \
    gcc/7.3.0 \
    openmpi/4.0.4-cuda102 
    
### Python requirements
    torch == 1.10.2
    numpy == 1.19.5
    scipy == 1.5.4
    FrEIA == 0.2
    nflows == 0.14
    wandb == 0.12.11
    matplotlib == 3.3.3
    horovod == 0.25.0
