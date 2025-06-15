# CNN
Implementation of a basic Convolutional Neural Network in C from scratch (no libraries) trained on the MNIST dataset (handwritten digits) achieving 90% accuracy. 

# Installation
To compile the CNN, use the following command:
```bash
gcc -Wall -Wextra -g -O3  main.c lib/import.c lib/convolution.c lib/pooling.c lib/dense.c lib/output.c lib/backprop.c -o cnn -lm
```
To run the CNN, use the following command:
```bash
./cnn
```

# Features
- Convolutional Layer (cross-correlation)
- Pooling Layer (2x2 max pooling)
- Dense Layer (fully connectd)
- Activation Function (softmax)
- Loss Function (cross-entropy)

# Results
To measure the performance of the CNN, we train it on the dataset : `./MNIST/train-images-idx3-ubyte` and `./MNIST/train-labels-idx1-ubyte` and then test it on another dataset (in order to avoid overfitting) : `./MNIST/t10k-images-idx3-ubyte` and `./MNIST/t10k-labels-idx1-ubyte` containing 10,000 images of handwritten digits.

When trained for 1 epoch, we get the following results with a learning rate of 0.005:
```bash
|----------------------------------------|
| Average Loss: 0.415566 | Accuracy: 86% |
|----------------------------------------|
```
When trained for 5 epochs, we get the following results with a learning rate of 0.005:
```bash
|----------------------------------------|
| Average Loss: 0.288800 | Accuracy: 91% |
|----------------------------------------|
```