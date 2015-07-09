# MNIST

An investigation into neural networks using the [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits.

## 1. MNIST Viewer

A simple page for viewing MNIST images.

### How to use

1. Download/unzip MNIST's "train-images-idx3-ubyte" into this directory.
2. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
3. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

## 2. Softmax Algorithm

An implementation of the softmax algorithm that can be used to classify MNIST images.

### How to use

1. Download/unzip the MNIST training/test data/labels (four files total) into this directory.
2. Test it out:

    > import softmax
    > train = softmax.training_examples(); test = softmax.testing_examples() # takes a few seconds
    > # train = train[:1000] # if you want it to finish faster
    > softmax.run(train, test) # takes quite a while

3. Feel free to poke around with the algorithm constants to see if you can reduce the error rate.
4. The incorrect predictions on the test data are stored for investigation in `softmax.incorrect_predictions`.

