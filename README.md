# GradiatorJS



GradiatorJS is a lightweight, from-scratch neural network library built in typescript which you can use to create and train deep neural nets directly in your browser. It is an autodiff engine with backpropagation which is implemented over a dynamically build DAG i.e. the library will automatically compute the gradients of all parameters with respect to a final output (like a loss fn). This allows you to define dynamic neural nets (including Fully Connected and Convolutional Neural Networks), train these networks on custom datasets using mini-batch gradient descent etc etc.

This project is done with a pedagogical intention in mind (to help concepts feel intuitive and easy to understand). You can use this library in your own projects using the npm package. 

Note: While this library can be used on its own, it is being used as the core engine for **[Visualizer](https://github.com/vark1/nn-visualizer)**, an interactive platform for building, training, and seeing inside your models in real-time.

## Features:

* **Dynamic Computation Graph:** At its core is a `Val` object that tracks all operations to build a computation graph on the fly, enabling automatic backpropagation. It can store anything from a scalar to an ND array.
* **Modern Network Layers:** Comes with all the essential layers needed for modern image classification tasks:
    * `Dense`: FC layer
    * `Conv2D`: Convolutional Layer
    * `MaxPooling2D`: Max Pool Layer
    * `Flatten`: Flattening Layer
* **Standard Activation Functions:** `ReLU`, `Sigmoid`, `Tanh`, `Softmax`.
* **Loss Functions:** Standard loss functions like `crossEntropyLoss_binary`, `crossEntropyLoss_categorical` and a numerically stable `crossEntropyLoss_softmax` for multi-class problems.
* **Accuracy Functions:** `calcBinaryAccuracy` and `calcMultiClassAccuracy`.
* **Operations:** `add`, `sub`, `mul`, `dot`, `pow`, `div`, `divElementWise`, `negate`, `abs`, `exp`, `log`, `sum`, `mean`, `conv2d`, `maxPool2d`. (all with broadcasting!)
* **Async training:** Ability to pause, resume, stop training using state management.
* **Mini-Batch GD:** `getMiniBatch` which creates shuffled batches.
* **Activation data for a sample batch:** While training, it passes activation data of a sample in callback, which can be used to visualize the  network's current state.
* **Pure TypeScript/JavaScript:** No external dependencies for the core engine, making it lightweight and easy to understand.

## Installation

Using npm:
```bash
npm install gradiatorjs
```

## Getting Started

### Creating a binary classification network

```typescript
import { Sequential, Dense, Flatten, relu, sigmoid, crossEntropyLoss_binary } from 'gradiatorjs';

const model = new Sequential(
    // Assuming an input shape of [batch, 64, 64, 3] for cat images
    new Flatten(),
    new Dense(12288, 20, relu),
    new Dense(20, 7, relu),
    new Dense(7, 5, relu),
    new Dense(5, 1, sigmoid)
);

const loss = crossEntropyLoss_binary(predictions, true_labels);
```

### Creating a Convolutional Network

```typescript
import { Sequential, Dense, Conv2D, MaxPooling2D, Flatten, relu, softmaxCrossEntropyLoss } from 'gradiatorjs';

const model = new Sequential(
    // Assuming an input shape of [batch, 28, 28, 1] for a dataset like MNIST
    new Conv(1, 6, 5, 1, 0, relu), // in_channels, out_channels, kernel, stride, padding, activation
    new MaxPooling2D(2, 2),
    new Conv(6, 16, 5, 1, 0, relu),
    new MaxPooling2D(2, 2),
    new Flatten(),
    new Dense(256, 120, relu),
    new Dense(120, 84, relu),
    new Dense(84, 10)
);

const loss = softmaxCrossEntropyLoss(logits, one_hot_labels);
```
