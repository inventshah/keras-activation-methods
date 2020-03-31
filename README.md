# Keras Activation Methods
Simple neural network for the fashion mnist to test activation methods.

### How to Use:
Install dependencies:

```
pip install numpy
pip install matplotlib
pip install tensorflow
pip install Keras
```

Clone: `git clone https://github.com/inventshah/keras-activation-methods.git`

Run: `python3 main.py`

### Results:
Used the fashion mnist dataset with the following base sequential model:

Layer | Output Shape | Param #
:----:|:------------:|:-------:
Flatten | (None, 784) | 0
Dense | (None, 32) | 25120
Dense | (None, 10) | 330

The following activation methods were used for the first dense layer. Each variation of the model trained for 10 epoches on the same training and testing set.

Activation | Accuracy 
:---------:|:--------:
Relu | 87.01%
Sigmoid | 86.20%
Linear | 83.59%
Exponential | 86.15%
Hard Sigmoid | 70.00%
Softplus | 87.25%
Softsign | 87.16%
Tanh | 87.08%

Relu, softplus, softsign, and tanh preformed similarly. While hard sigmoid had the worst accuracy.

See the `models/` folder for the final models of each variation and `images/` folder for graphs of each methods improvement.

### Built With:
- Python 3
- TensorFlow