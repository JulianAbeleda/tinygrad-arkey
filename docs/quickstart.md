# Quick Start Guide

This guide assumes no prior knowledge of pytorch or any other deep learning framework, but does assume some basic knowledge of neural networks.
It is intended to be a very quick overview of the high level API that tinygrad provides.

This guide is also structured as a tutorial which at the end of it you will have a working model that can classify handwritten digits.

We need some imports to get started:

```python
import numpy as np
from tinygrad.helpers import Timing
```

## Tensors

Tensors are the base data structure in tinygrad. They can be thought of as a multidimensional array of a specific data type.
All high level operations in tinygrad operate on these tensors.

The tensor class can be imported like so:

```python
from tinygrad import Tensor
```

Tensors can be created from an existing data structure like a python list or numpy ndarray:

```python
t1 = Tensor([1, 2, 3, 4, 5])
na = np.array([1, 2, 3, 4, 5])
t2 = Tensor(na)
```

Tensors can also be created using one of the many factory methods:

```python
full = Tensor.full(shape=(2, 3), fill_value=5) # create a tensor of shape (2, 3) filled with 5
zeros = Tensor.zeros(2, 3) # create a tensor of shape (2, 3) filled with 0
ones = Tensor.ones(2, 3) # create a tensor of shape (2, 3) filled with 1

full_like = Tensor.full_like(full, fill_value=2) # create a tensor of the same shape as `full` filled with 2
zeros_like = Tensor.zeros_like(full) # create a tensor of the same shape as `full` filled with 0
ones_like = Tensor.ones_like(full) # create a tensor of the same shape as `full` filled with 1

eye = Tensor.eye(3) # create a 3x3 identity matrix
arange = Tensor.arange(start=0, stop=10, step=1) # create a tensor of shape (10,) filled with values from 0 to 9

rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
randn = Tensor.randn(2, 3) # create a tensor of shape (2, 3) filled with random values from a standard normal distribution
uniform = Tensor.uniform(2, 3, low=0, high=10) # create a tensor of shape (2, 3) filled with random values from a uniform distribution between 0 and 10
```

There are even more of these factory methods, you can find them in the [Tensor Creation](tensor/creation.md) file.

All the tensors creation methods can take a `dtype` argument to specify the data type of the tensor, find the supported `dtype` in [dtypes](dtypes.md).

```python
from tinygrad import dtypes

t3 = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)
```

Tensors allow you to perform operations on them like so:

```python
t4 = Tensor([1, 2, 3, 4, 5])
t5 = (t4 + 1) * 2
t6 = (t5 * t4).relu().log_softmax()
```

All of these operations are lazy and are only executed when you realize the tensor using `.realize()` or `.numpy()`.

```python
print(t6.numpy())
# [-56. -48. -36. -20.   0.]
```

There are a lot more operations that can be performed on tensors, you can find them in the [Tensor Ops](tensor/ops.md) file.
Additionally reading through [abstractions2.py](https://github.com/JulianAbeleda/tinygrad-arkey/blob/master/docs/abstractions2.py) will help you understand how operations on these tensors make their way down to your hardware.

## Models

Neural networks in tinygrad are really just represented by the operations performed on tensors.
These operations are commonly grouped into the `__call__` method of a class which allows modularization and reuse of these groups of operations.
These classes do not need to inherit from any base class, in fact if they don't need any trainable parameters they don't even need to be a class!

An example of this would be the `nn.Linear` class which represents a linear layer in a neural network.

```python
class Linear:
  def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
    self.weight = getattr(Tensor, initialization)(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    return x.linear(self.weight.transpose(), self.bias)
```

There are more neural network modules already implemented in [nn](nn.md), and you can also implement your own.

We will be implementing a simple neural network that can classify handwritten digits from the MNIST dataset.
Our classifier will be a simple 2 layer neural network with a Leaky ReLU activation function.
It will use a hidden layer size of 128 and an output layer size of 10 (one for each digit) with no bias on either Linear layer.

```python
class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leaky_relu()
    x = self.l2(x)
    return x

net = TinyNet()
```

We can see that the forward pass of our neural network is just the sequence of operations performed on the input tensor `x`.
We can also see that functional operations like `leaky_relu` are not defined as classes and instead are just methods we can just call.
Finally, we initialize an instance of our neural network, and we are ready for inference.

## Inference-only note

Training APIs that rely on autograd and `tinygrad.nn.optim` have been removed in this branch.
This guide now focuses on inference-only workflows.
Load your model data (or dataset loader output) and call the model directly to run forward passes.

## Evaluation

Now that we have our neural network we can evaluate it on a dataset split.
We will be using the MNIST dataset loader in [/extra/datasets](https://github.com/JulianAbeleda/tinygrad-arkey/blob/master/extra/datasets) and evaluate 1000 random batches of size 64.

```python
from extra.datasets import fetch_mnist

_, _, X_test, Y_test = fetch_mnist()
```

```python
with Timing("Time: "):
  avg_acc = 0
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp])
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / 1000}")
```

## And that's it

Highly recommend you check out the [examples/](https://github.com/JulianAbeleda/tinygrad-arkey/blob/master/examples) folder for more examples of using tinygrad.
Reading the source code of tinygrad is also a great way to learn how it works.
Specifically the tests in [test/](https://github.com/JulianAbeleda/tinygrad-arkey/blob/master/test) are a great place to see how to use and the semantics of the different operations.
There are also a bunch of models implemented in [models/](https://github.com/JulianAbeleda/tinygrad-arkey/blob/master/extra/models) that you can use as a reference.

Additionally, feel free to ask questions in the `#learn-tinygrad` channel on the [discord](https://discord.gg/beYbxwxVdx). Don't ask to ask, just ask!

## Extras

### JIT

Additionally, it is possible to speed up the computation of certain neural networks by using the JIT.
Currently, this does not support models with varying input sizes and non tinygrad operations.

To use the JIT we just need to add a function decorator to the forward pass of our neural network and ensure that the input and output are realized tensors.
Or in this case we will create a wrapper function and decorate the wrapper function to speed up the evaluation of our neural network.

```python
from tinygrad import TinyJit

@TinyJit
def jit(x):
  return net(x).realize()

with Timing("Time: "):
  avg_acc = 0
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp])
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass with jit
    out = jit(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / 1000}")
```

You will find that the evaluation time is much faster than before and that your accelerator utilization is much higher.

### Saving and Loading Models

The standard weight format for tinygrad is [safetensors](https://github.com/huggingface/safetensors). This means that you can load the weights of any model also using safetensors into tinygrad.
There are functions in [state.py](https://github.com/JulianAbeleda/tinygrad-arkey/blob/master/tinygrad/nn/state.py) to save and load models to and from this format.

```python
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

# first we need the state dict of our model
state_dict = get_state_dict(net)

# then we can just save it to a file
safe_save(state_dict, "model.safetensors")

# and load it back in
state_dict = safe_load("model.safetensors")
load_state_dict(net, state_dict)
```

Many of the models in the [models/](https://github.com/JulianAbeleda/tinygrad-arkey/tree/master/extra/models) folder have a `load_from_pretrained` method that will download and load the weights for you. These usually are pytorch weights meaning that you would need pytorch installed to load them.

### Environment Variables

There exist a bunch of environment variables that control the runtime behavior of tinygrad.
Some of the commons ones are `DEBUG` and the different backend enablement variables.

You can find a full list and their descriptions in [env_vars.md](env_vars.md).

### Visualizing the Computation Graph

It is possible to visualize the computation graph of a neural network using VIZ=1.
