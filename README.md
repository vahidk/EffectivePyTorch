# Effective PyTorch

Table of Contents
=================
## Part I: PyTorch Fundamentals
1.  [PyTorch basics](#basics)
2.  [Encapsulate your model with Modules](#modules)
3.  [Broadcasting the good and the ugly](#broadcast)
4.  [Take advantage of the overloaded operators](#overloaded_ops)
5.  [Optimizing runtime with TorchScript](#torchscript)
6.  [Building efficient custom data loaders](#dataloader)
7.  [Numerical stability in PyTorch](#stable)
8.  [Faster training with automatic mixed precision](#amp)
---

_To install PyTorch follow the [instructions on the official website](https://pytorch.org/):_
```
pip install torch torchvision
```

_We aim to gradually expand this series by adding new articles and keep the content up to date with the latest releases of PyTorch API. If you have suggestions on how to improve this series or find the explanations ambiguous, feel free to create an issue, send patches, or reach out by email._

# Part I: PyTorch Fundamentals
<a name="fundamentals"></a>

## PyTorch basics
<a name="basics"></a>
PyTorch is one of the most popular libraries for numerical computation and currently is amongst the most widely used libraries for performing machine learning research. In many ways PyTorch is similar to NumPy, with the additional benefit that PyTorch allows you to perform your computations on CPUs, GPUs, and TPUs without any material change to your code. PyTorch also makes it easy to distribute your computation across multiple devices or machines. One of the most important features of PyTorch is automatic differentiation. It allows computing the gradients of your functions analytically in an efficient manner which is crucial for training machine learning models using gradient descent method. Our goal here is to provide a gentle introduction to PyTorch and discuss best practices for using PyTorch.

The first thing to learn about PyTorch is the concept of Tensors. Tensors are simply multidimensional arrays. A PyTorch Tensor is very similar to a NumPy array with some ~~magical~~ additional functionality. 

A tensor can store a scalar value:
```python
import torch
a = torch.tensor(3)
print(a)  # tensor(3)
```

or an array:
```python
b = torch.tensor([1, 2])
print(b)  # tensor([1, 2])
```

a matrix:
```python
c = torch.zeros([2, 2])
print(c)  # tensor([[0., 0.], [0., 0.]])
```

or any arbitrary dimensional tensor:
```python
d = torch.rand([2, 2, 2])
```

Tensors can be used to perform algebraic operations efficiently. One of the most commonly used operations in machine learning applications is matrix multiplication. Say you want to multiply two random matrices of size 3x5 and 5x4, this can be done with the matrix multiplication (@) operation:
```python
import torch

x = torch.randn([3, 5])
y = torch.randn([5, 4])
z = x @ y

print(z)
```

Similarly, to add two vectors, you can do:
```python
z = x + y
```

To convert a tensor into a numpy array you can call Tensor's numpy() method:
```python
print(z.numpy())
```

And you can always convert a numpy array into a tensor by:
```python
x = torch.tensor(np.random.normal([3, 5]))
```

### Automatic differentiation

The most important advantage of PyTorch over NumPy is its automatic differentiation functionality which is very useful in optimization applications such as optimizing parameters of a neural network. Let's try to understand it with an example.

Say you have a composite function which is a chain of two functions: `g(u(x))`.
To compute the derivative of `g` with respect to `x` we can use the chain rule which states that: `dg/dx = dg/du * du/dx`. PyTorch can analytically compute the derivatives for us.

To compute the derivatives in PyTorch first we create a tensor and set its `requires_grad` to true. We can use tensor operations to define our functions. We assume `u` is a quadratic function and `g` is a simple linear function:
```python
x = torch.tensor(1.0, requires_grad=True)

def u(x):
  return x * x

def g(u):
  return -u
```

In this case our composite function is `g(u(x)) = -x*x`. So its derivative with respect to `x` is `-2x`. At point `x=1`, this is equal to `-2`.

Let's verify this. This can be done using grad function in PyTorch:
```python
dgdx = torch.autograd.grad(g(u(x)), x)[0]
print(dgdx)  # tensor(-2.)
```

### Curve fitting

To understand how powerful automatic differentiation can be let's have a look at another example. Assume that we have samples from a curve (say `f(x) = 5x^2 + 3`) and we want to estimate `f(x)` based on these samples. We define a parametric function `g(x, w) = w0 x^2 + w1 x + w2`, which is a function of the input `x` and latent parameters `w`, our goal is then to find the latent parameters such that `g(x, w) ≈ f(x)`. This can be done by minimizing the following loss function: `L(w) = Σ (f(x) - g(x, w))^2`. Although there's a closed form solution for this simple problem, we opt to use a more general approach that can be applied to any arbitrary differentiable function, and that is using stochastic gradient descent. We simply compute the average gradient of `L(w)` with respect to `w` over a set of sample points and move in the opposite direction.

Here's how it can be done in PyTorch:

```python
import numpy as np
import torch

# Assuming we know that the desired function is a polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients and initialize it with
# random noise.
w = torch.tensor(torch.randn([3, 1]), requires_grad=True)

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
opt = torch.optim.Adam([w], 0.1)

def model(x):
    # We define yhat to be our estimate of y.
    f = torch.stack([x * x, x, torch.ones_like(x)], 1)
    yhat = torch.squeeze(f @ w, 1)
    return yhat

def compute_loss(y, yhat):
    # The loss is defined to be the mean squared error distance between our
    # estimate of y and its true value. 
    loss = torch.nn.functional.mse_loss(yhat, y)
    return loss

def generate_data():
    # Generate some training data based on the true function
    x = torch.rand(100) * 20 - 10
    y = 5 * x * x + 3
    return x, y

def train_step():
    x, y = generate_data()

    yhat = model(x)
    loss = compute_loss(y, yhat)

    opt.zero_grad()
    loss.backward()
    opt.step()

for _ in range(1000):
    train_step()

print(w.detach().numpy())
```
By running this piece of code you should see a result close to this:
```python
[4.9924135, 0.00040895029, 3.4504161]
```
Which is a relatively close approximation to our parameters.

This is just tip of the iceberg for what PyTorch can do. Many problems such as optimizing large neural networks with millions of parameters can be implemented efficiently in PyTorch in just a few lines of code. PyTorch takes care of scaling across multiple devices, and threads, and supports a variety of platforms.

## Encapsulate your model with Modules
<a name="modules"></a>
In the previous example we used bare bone tensors and tensor operations to build our model. To make your code slightly more organized it's recommended to use PyTorch's modules. A module is simply a container for your parameters and encapsulates model operations. For example say you want to represent a linear model `y = ax + b`. This model can be represented with the following code:

```python
import torch

class Net(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.a = torch.nn.Parameter(torch.rand(1))
    self.b = torch.nn.Parameter(torch.rand(1))

  def forward(self, x):
    yhat = self.a * x + self.b
    return yhat
```

To use this model in practice you instantiate the module and simply call it like a function:
```python
x = torch.arange(100, dtype=torch.float32)

net = Net()
y = net(x)
```

Parameters are essentially tensors with `requires_grad` set to true. It's convenient to use parameters because you can simply retrieve them all with module's `parameters()` method:
```python
for p in net.parameters():
    print(p)
```

Now, say you have an unknown function `y = 5x + 3 + some noise`, and you want to optimize the parameters of your model to fit this function.  You can start by sampling some points from your function:
```python
x = torch.arange(100, dtype=torch.float32) / 100
y = 5 * x + 3 + torch.rand(100) * 0.3
```

Similar to the previous example, you can define a loss function and optimize the parameters of your model as follows:
```python
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for i in range(10000):
  net.zero_grad()
  yhat = net(x)
  loss = criterion(yhat, y)
  loss.backward()
  optimizer.step()

print(net.a, net.b) # Should be close to 5 and 3
```

PyTorch comes with a number of predefined modules. One such module is `torch.nn.Linear` which is a more general form of a linear function than what we defined above. We can rewrite our module above using `torch.nn.Linear` like this:

```python
class Net(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(1, 1)

  def forward(self, x):
    yhat = self.linear(x.unsqueeze(1)).squeeze(1)
    return yhat
```

Note that we used squeeze and unsqueeze since `torch.nn.Linear` operates on batch of vectors as opposed to scalars.

By default calling parameters() on a module will return the parameters of all its submodules:
```python
net = Net()
for p in net.parameters():
    print(p)
```

There are some predefined modules that act as a container for other modules. The most commonly used container module is `torch.nn.Sequential`. As its name implies it's used to to stack multiple modules (or layers) on top of each other. For example to stack two Linear layers with a `ReLU` nonlinearity in between you can do:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10),
)
```

## Broadcasting the good and the ugly
<a name="broadcast"></a>
PyTorch supports broadcasting elementwise operations. Normally when you want to perform operations like addition and multiplication, you need to make sure that shapes of the operands match, e.g. you can’t add a tensor of shape `[3, 2]` to a tensor of shape `[3, 4]`. But there’s a special case and that’s when you have a singular dimension. PyTorch implicitly tiles the tensor across its singular dimensions to match the shape of the other operand. So it’s valid to add a tensor of shape `[3, 2]` to a tensor of shape `[3, 1]`.

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[1.], [2.]])
# c = a + b.repeat([1, 2])
c = a + b

print(c)
```

Broadcasting allows us to perform implicit tiling which makes the code shorter, and more memory efficient, since we don’t need to store the result of the tiling operation. One neat place that this can be used is when combining features of varying length. In order to concatenate features of varying length we commonly tile the input tensors, concatenate the result and apply some nonlinearity. This is a common pattern across a variety of neural network architectures:

```python
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])

linear = torch.nn.Linear(11, 10)

# concat a and b and apply nonlinearity
tiled_b = b.repeat([1, 3, 1])
c = torch.cat([a, tiled_b], 2)
d = torch.nn.functional.relu(linear(c))

print(d.shape)  # torch.Size([5, 3, 10])
```

But this can be done more efficiently with broadcasting. We use the fact that `f(m(x + y))` is equal to `f(mx + my)`. So we can do the linear operations separately and use broadcasting to do implicit concatenation:

```python
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])

linear1 = torch.nn.Linear(5, 10)
linear2 = torch.nn.Linear(6, 10)

pa = linear1(a)
pb = linear2(b)
d = torch.nn.functional.relu(pa + pb)

print(d.shape)  # torch.Size([5, 3, 10])
```

In fact this piece of code is pretty general and can be applied to tensors of arbitrary shape as long as broadcasting between tensors is possible:

```python
class Merge(torch.nn.Module):
    def __init__(self, in_features1, in_features2, out_features, activation=None):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features1, out_features)
        self.linear2 = torch.nn.Linear(in_features2, out_features)
        self.activation = activation

    def forward(self, a, b):
        pa = self.linear1(a)
        pb = self.linear2(b)
        c = pa + pb
        if self.activation is not None:
            c = self.activation(c)
        return c
```

So far we discussed the good part of broadcasting. But what’s the ugly part you may ask? Implicit assumptions almost always make debugging harder to do. Consider the following example:

```python
a = torch.tensor([[1.], [2.]])
b = torch.tensor([1., 2.])
c = torch.sum(a + b)

print(c)
```

What do you think the value of `c` would be after evaluation? If you guessed 6, that’s wrong. It’s going to be 12. This is because when rank of two tensors don’t match, PyTorch automatically expands the first dimension of the tensor with lower rank before the elementwise operation, so the result of addition would be `[[2, 3], [3, 4]]`, and the reducing over all parameters would give us 12.

The way to avoid this problem is to be as explicit as possible. Had we specified which dimension we would want to reduce across, catching this bug would have been much easier:

```python
a = torch.tensor([[1.], [2.]])
b = torch.tensor([1., 2.])
c = torch.sum(a + b, 0)

print(c)
```

Here the value of `c` would be `[5, 7]`, and we immediately would guess based on the shape of the result that there’s something wrong. A general rule of thumb is to always specify the dimensions in reduction operations and when using `torch.squeeze`.

## Take advantage of the overloaded operators
<a name="overloaded_ops"></a>
Just like NumPy, PyTorch overloads a number of python operators to make PyTorch code shorter and more readable.

The slicing op is one of the overloaded operators that can make indexing tensors very easy:
```python
z = x[begin:end]  # z = torch.narrow(0, begin, end-begin)
```
Be very careful when using this op though. The slicing op, like any other op, has some overhead. Because it's a common op and innocent looking it may get overused a lot which may lead to inefficiencies. To understand how inefficient this op can be let's look at an example. We want to manually perform reduction across the rows of a matrix:
```python
import torch
import time

x = torch.rand([500, 10])

z = torch.zeros([10])

start = time.time()
for i in range(500):
    z += x[i]
print("Took %f seconds." % (time.time() - start))
```
This runs quite slow and the reason is that we are calling the slice op 500 times, which adds a lot of overhead. A better choice would have been to use `torch.unbind` op to slice the matrix into a list of vectors all at once:
```python
z = torch.zeros([10])
for x_i in torch.unbind(x):
    z += x_i
```
This is significantly (~30% on my machine) faster. 

Of course, the right way to do this simple reduction is to use `torch.sum` op to this in one op:
```python
z = torch.sum(x, dim=0)
```
which is extremely fast (~100x faster on my machine).

PyTorch also overloads a range of arithmetic and logical operators:
```python
z = -x  # z = torch.neg(x)
z = x + y  # z = torch.add(x, y)
z = x - y
z = x * y  # z = torch.mul(x, y)
z = x / y  # z = torch.div(x, y)
z = x // y
z = x % y
z = x ** y  # z = torch.pow(x, y)
z = x @ y  # z = torch.matmul(x, y)
z = x > y
z = x >= y
z = x < y
z = x <= y
z = abs(x)  # z = torch.abs(x)
z = x & y
z = x | y
z = x ^ y  # z = torch.logical_xor(x, y)
z = ~x  # z = torch.logical_not(x)
z = x == y  # z = torch.eq(x, y)
z = x != y  # z = torch.ne(x, y)
```

You can also use the augmented version of these ops. For example `x += y` and `x **= 2` are also valid.

Note that Python doesn't allow overloading `and`, `or`, and `not` keywords.


## Optimizing runtime with TorchScript
<a name="torchscript"></a>
PyTorch is optimized to perform operations on large tensors. Doing many operations on small tensors is quite inefficient in PyTorch. So, whenever possible you should rewrite your computations in batch form to reduce overhead and improve performance. If there's no way you can manually batch your operations, using TorchScript may improve your code's performance. TorchScript is simply a subset of Python functions that are recognized by PyTorch. PyTorch can automatically optimize your TorchScript code using its just in time (jit) compiler and reduce some overheads.

Let's look at an example. A very common operation in ML applications is "batch gather". This operation can simply written as `output[i] = input[i, index[i]]`. This can be simply implemented in PyTorch as follows:
```python
import torch
def batch_gather(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)
```

To implement the same function using TorchScript simply use the `torch.jit.script` decorator:
```python
@torch.jit.script
def batch_gather_jit(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)
```
On my tests this is about 10% faster.

But nothing beats manually batching your operations. A vectorized implementation in my tests is 100 times faster:
```python
def batch_gather_vec(tensor, indices):
    shape = list(tensor.shape)
    flat_first = torch.reshape(
        tensor, [shape[0] * shape[1]] + shape[2:])
    offset = torch.reshape(
        torch.arange(shape[0]).cuda() * shape[1],
        [shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices + offset]
    return output
```

## Building efficient custom data loaders
<a name="dataloader"></a>

In the last lesson we talked about writing efficient PyTorch code. But to make your code run with maximum efficiency you also need to load your data efficiently into your device's memory. Fortunately PyTorch offers a tool to make data loading easy. It's called a `DataLoader`. A `DataLoader` uses multiple workers to simultanously load data from a `Dataset` and optionally uses a `Sampler` to sample data entries and form a batch.

If you can randomly access your data, using a `DataLoader` is very easy: You simply need to implement a `Dataset` class that implements `__getitem__` (to read each data item) and `__len__` (to return the number of items in the dataset) methods. For example here's how to load images from a given directory:

```python
import glob
import os
import random
import cv2
import torch

class ImageDirectoryDataset(torch.utils.data.Dataset):
    def __init__(path, pattern):
        self.paths = list(glob.glob(os.path.join(path, pattern)))

    def __len__(self):
        return len(self.paths)

    def __item__(self):
        path = random.choice(paths)
        return cv2.imread(path, 1)
```

To load all jpeg images from a given directory you can then do the following:
```python
dataloader = torch.utils.data.DataLoader(ImageDirectoryDataset("/data/imagenet/*.jpg"), num_workers=8)
for data in dataloader:
    # do something with data
```

Here we are using 8 workers to simultanously read our data from the disk. You can tune the number of workers on your machine for optimal results.

Using a `DataLoader` to read data with random access may be ok if you have fast storage or if your data items are large. But imagine having a network file system with slow connection. Requesting individual files this way can be extremely slow and would probably end up becoming the bottleneck of your training pipeline.

A better approach is to store your data in a contiguous file format which can be read sequentially. For example if you have a large collection of images you can use tar to create a single archive and extract files from the archive sequentially in python. To do this you can use PyTorch's `IterableDataset`. To create an `IterableDataset` class you only need to implement an `__iter__` method which sequentially reads and yields data items from the dataset.

A naive implementation would like this:

```python
import tarfile
import torch

def tar_image_iterator(path):
    tar = tarfile.open(self.path, "r")
    for tar_info in tar:
        file = tar.extractfile(tar_info)
        content = file.read()
        yield cv2.imdecode(content, 1)
        file.close()
        tar.members = []
    tar.close()

class TarImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __iter__(self):
        yield from tar_image_iterator(self.path)
```

But there's a major problem with this implementation. If you try to use DataLoader to read from this dataset with more than one worker you'd observe a lot of duplicated images:

```python
dataloader = torch.utils.data.DataLoader(TarImageDataset("/data/imagenet.tar"), num_workers=8)
for data in dataloader:
    # data contains duplicated items
```

The problem is that each worker creates a separate instance of the dataset and each would start from the beginning of the dataset. One way to avoid this is to instead of having one tar file, split your data into `num_workers` separate tar files and load each with a separate worker:

```python
class TarImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # For simplicity we assume num_workers is equal to number of tar files
        if worker_info is None or worker_info.num_workers != len(self.paths):
            raise ValueError("Number of workers doesn't match number of files.")
        yield from tar_image_iterator(self.paths[worker_info.worker_id])
```

This is how our dataset class can be used:
```python
dataloader = torch.utils.data.DataLoader(
    TarImageDataset(["/data/imagenet_part1.tar", "/data/imagenet_part2.tar"]), num_workers=2)
for data in dataloader:
    # do something with data
```

We discussed a simple strategy to avoid duplicated entries problem. [tfrecord](https://github.com/vahidk/tfrecord) package uses slightly more sophisticated strategies to shard your data on the fly.

## Numerical stability in PyTorch
<a name="stable"></a>
When using any numerical computation library such as NumPy or PyTorch, it's important to note that writing mathematically correct code doesn't necessarily lead to correct results. You also need to make sure that the computations are stable.

Let's start with a simple example. Mathematically, it's easy to see that `x * y / y = x` for any non zero value of `x`. But let's see if that's always true in practice:
```python
import numpy as np

x = np.float32(1)

y = np.float32(1e-50)  # y would be stored as zero
z = x * y / y

print(z)  # prints nan
```

The reason for the incorrect result is that `y` is simply too small for float32 type. A similar problem occurs when `y` is too large:

```python
y = np.float32(1e39)  # y would be stored as inf
z = x * y / y

print(z)  # prints nan
```

The smallest positive value that float32 type can represent is 1.4013e-45 and anything below that would be stored as zero. Also, any number beyond 3.40282e+38, would be stored as inf.

```python
print(np.nextafter(np.float32(0), np.float32(1)))  # prints 1.4013e-45
print(np.finfo(np.float32).max)  # print 3.40282e+38
```

To make sure that your computations are stable, you want to avoid values with small or very large absolute value. This may sound very obvious, but these kind of problems can become extremely hard to debug especially when doing gradient descent in PyTorch. This is because you not only need to make sure that all the values in the forward pass are within the valid range of your data types, but also you need to make sure of the same for the backward pass (during gradient computation).

Let's look at a real example. We want to compute the softmax over a vector of logits. A naive implementation would look something like this:
```python
import torch

def unstable_softmax(logits):
    exp = torch.exp(logits)
    return exp / torch.sum(exp)

print(unstable_softmax(torch.tensor([1000., 0.])).numpy())  # prints [ nan, 0.]
```
Note that computing the exponential of logits for relatively small numbers results to gigantic results that are out of float32 range. The largest valid logit for our naive softmax implementation is `ln(3.40282e+38) = 88.7`, anything beyond that leads to a nan outcome.

But how can we make this more stable? The solution is rather simple. It's easy to see that `exp(x - c) Σ exp(x - c) = exp(x) / Σ exp(x)`. Therefore we can subtract any constant from the logits and the result would remain the same. We choose this constant to be the maximum of logits. This way the domain of the exponential function would be limited to `[-inf, 0]`, and consequently its range would be `[0.0, 1.0]` which is desirable:

```python
import torch

def softmax(logits):
    exp = torch.exp(logits - torch.reduce_max(logits))
    return exp / torch.sum(exp)

print(softmax(torch.tensor([1000., 0.])).numpy())  # prints [ 1., 0.]
```

Let's look at a more complicated case. Consider we have a classification problem. We use the softmax function to produce probabilities from our logits. We then define our loss function to be the cross entropy between our predictions and the labels. Recall that cross entropy for a categorical distribution can be simply defined as `xe(p, q) = -Σ p_i log(q_i)`. So a naive implementation of the cross entropy would look like this:

```python
def unstable_softmax_cross_entropy(labels, logits):
    logits = torch.log(softmax(logits))
    return -torch.sum(labels * logits)

labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])

xe = unstable_softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints inf
```

Note that in this implementation as the softmax output approaches zero, the log's output approaches infinity which causes instability in our computation. We can rewrite this by expanding the softmax and doing some simplifications:

```python
def softmax_cross_entropy(labels, logits, dim=-1):
    scaled_logits = logits - torch.max(logits)
    normalized_logits = scaled_logits - torch.logsumexp(scaled_logits, dim)
    return -torch.sum(labels * normalized_logits)

labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])

xe = softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints 500.0
```

We can also verify that the gradients are also computed correctly:
```python
logits.requires_grad_(True)
xe = softmax_cross_entropy(labels, logits)
g = torch.autograd.grad(xe, logits)[0]
print(g.numpy())  # prints [0.5, -0.5]
```

Let me remind again that extra care must be taken when doing gradient descent to make sure that the range of your functions as well as the gradients for each layer are within a valid range. Exponential and logarithmic functions when used naively are especially problematic because they can map small numbers to enormous ones and the other way around.


## Faster training with mixed precision
<a name="amp"></a>
By default tensors and model parameters in PyTorch are stored in 32-bit floating point precision. Training neural networks using 32-bit floats is usually stable and doesn&#39;t cause major numerical issues, however neural networks have been shown to perform quite well in 16-bit and even lower precisions. Computation in lower precisions can be significantly faster on modern GPUs. It also has the extra benefit of using less memory enabling training larger models and/or with larger batch sizes which can boost the performance further. The problem though is that training in 16 bits often becomes very unstable because the precision is usually not enough to perform some operations like accumulations.

To help with this problem PyTorch supports training in mixed precision. In a nutshell mixed-precision training is done by performing some expensive operations (like convolutions and matrix multplications) in 16-bit by casting down the inputs while performing other numerically sensitive operations like accumulations in 32-bit. This way we get all the benefits of 16-bit computation without its drawbacks. Next we talk about using Autocast and GradScaler to do automatic mixed-precision training.

### Autocast

`autocast` helps improve runtime performance by automatically casting down data to 16-bit for some computations. To understand how it works let&#39;s look at an example:

```python
import torch

x = torch.rand([32, 32]).cuda()
y = torch.rand([32, 32]).cuda()

with torch.cuda.amp.autocast():
  a = x + y
  b = x @ y
print(a.dtype)  # prints torch.float32
print(b.dtype)  # prints torch.float16
```

Note both `x` and `y` are 32-bit tensors, but `autocast` performs matrix multiplication in 16-bit while keeping addition operation in 32-bit. What if one of the operands is in 16-bit?

```python
import torch

x = torch.rand([32, 32]).cuda()
y = torch.rand([32, 32]).cuda().half()

with torch.cuda.amp.autocast():
  a = x + y
  b = x @ y
print(a.dtype)  # prints torch.float32
print(b.dtype)  # prints torch.float16
```

Again `autocast` and casts down the 32-bit operand to 16-bit to perform matrix multiplication, but it doesn&#39;t change the addition operation. By default, addition of two tensors in PyTorch results in a cast to higher precision.

In practice, you can trust `autocast` to do the right casting to improve runtime efficiency. The important thing is to keep all your forward pass computations under `autocast` context:

```python
model = ...
loss_fn = ...

with torch.cuda.amp.autocast():
  outputs = model(inputs)
  loss = loss_fn(outputs, targets)
```

This maybe all you need if you have a relatively stable optimization problem and if you use a relatively low learning rate. Adding this one line of extra code can reduce your training up to half on modern hardware.

### GradScalar

As we mentioned in the beginning of this section, 16-bit precision may not always be enough for some computations. One particular case of interest is representing gradient values, a great portion of which are usually small values. Representing them with 16-bit floats often leads to buffer underflows (i.e. they&#39;d be represented as zeros). This makes training neural networks very unstable. `GradScalar` is designed to resolve this issue. It takes as input your loss value and multiplies it by a large scalar, inflating gradient values, and therefore making them represnetable in 16-bit precision. It then scales them down during gradient update to ensure parameters are updated correctly. This is generally what `GradScalar` does. But under the hood `GradScalar` is a bit smarter than that. Inflating the gradients may actually result in overflows which is equally bad. So `GradScalar` actually monitors the gradient values and if it detects overflows it skips updates, scaling down the scalar factor according to a configurable schedule. (The default schedule usually works but you may need to adjust that for your use case.)

Using `GradScalar` is very easy in practice:

```python
scaler = torch.cuda.amp.GradScaler()

loss = ...
optimizer = ...  # an instance torch.optim.Optimizer

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Note that we first create an instance of `GradScalar`. In training loop we call `GradScalar.scale` to scale the loss before calling backward to produce inflated gradients, we then use `GradScalar.step` which (may) update the model parameters. We then call `GradScalar.update` which performs the scalar update if needed. That&#39;s all!

The following is a sample code that show cases mixed precision training on a synthetic problem of learning to generate a checkerboard from image coordinates. You can paste it on a [Google Colab](https://colab.research.google.com/), set the backend to GPU and compare the single and mixed-precision performance. Note that this is a small toy example, in practice with larger networks you may see larger boosts in performance using mixed precision.

### An Example

### Generating a checker board

```python
import torch
import matplotlib.pyplot as plt
import time

def grid(width, height):
  hrange = torch.arange(width).unsqueeze(0).repeat([height, 1]).div(width)
  vrange = torch.arange(height).unsqueeze(1).repeat([1, width]).div(height)
  output = torch.stack([hrange, vrange], 0)
  return output


def checker(width, height, freq):
  hrange = torch.arange(width).reshape([1, width]).mul(freq / width / 2.0).fmod(1.0).gt(0.5)
  vrange = torch.arange(height).reshape([height, 1]).mul(freq / height / 2.0).fmod(1.0).gt(0.5)
  output = hrange.logical_xor(vrange).float()
  return output

# Note the inputs are grid coordinates and the target is a checkerboard
inputs = grid(512, 512).unsqueeze(0).cuda()
targets = checker(512, 512, 8).unsqueeze(0).unsqueeze(1).cuda()
```

### Defining a convolutional neural network

```python
class Net(torch.jit.ScriptModule):
  def __init__(self):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Conv2d(2, 256, 1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 1),
      torch.nn.BatchNorm2d(256),
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 1, 1))

  @torch.jit.script_method
  def forward(self, x):
    return self.net(x)
```

### Single precision training
```python
net = Net().cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), 0.001)

start_time = time.time()

for i in range(500):
  opt.zero_grad()
  outputs = net(inputs)
  loss = loss_fn(outputs, targets)
  loss.backward()
  opt.step()
print(loss)

print(time.time() - start_time)

plt.subplot(1,2,1); plt.imshow(outputs.squeeze().detach().cpu());
plt.subplot(1,2,2); plt.imshow(targets.squeeze().cpu()); plt.show()
```

### Mixed precision training
```python
net = Net().cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), 0.001)

scaler = torch.cuda.amp.GradScaler()

start_time = time.time()

for i in range(500):
  opt.zero_grad()
  with torch.cuda.amp.autocast():
    outputs = net(inputs)
    loss = loss_fn(outputs, targets)
  scaler.scale(loss).backward()
  scaler.step(opt)
  scaler.update()
print(loss)

print(time.time() - start_time)

plt.subplot(1,2,1); plt.imshow(outputs.squeeze().detach().cpu().float());
plt.subplot(1,2,2); plt.imshow(targets.squeeze().cpu().float()); plt.show()
```


### Reference
- https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
