# Neural networks from scratch
This repository implements some neural networks from scratch using python programming language. Currently master branch contains multi layer perceptron (MLP) neural network which trains by backprobagation algorithm using gradient descent optimizer and convolutional neural networks based on keras module. In the GPU branch, we try to use benefits of nvidia's cuda toolkit to accelerate the learning process. 

# How to install
We recommend to use a virtual environment to install this package. This is a simple guide to install this package.

    git clone https://github.com/alirezaafzalaghaei/neural-netwroks-from-scratch.git
    cd module
    pip instal pipenv
    pipenv install
    pipenv install -e .
    cd examples
    pipenv run python iris_test.py 

# Results
see `main.pdf` file in report folder for detailed information about this module.

## Todo
 - Porting MLP for Nvidia GPUs. 
