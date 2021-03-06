# MNIST

An investigation into neural networks using the [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits.

Two machine learning algorithms have been implemented: softmax (multiclass logistic regression) and multilayer perceptron (MLP, a kind of neural network).

Currently, the error rates for softmax and MLP are respectively about 10% and 3%.

## MNIST Viewer

A simple page for viewing MNIST images.

1. Download/unzip MNIST's "t10k-images-idx3-ubyte" into this directory.
2. Create a soft link "test-images" for this file: `ln -s t10k-images-idx3-ubyte test-images`
3. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
4. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

## Trying out the algorithms

1. Download/unzip the MNIST data (four files total) into this directory.
2. Create soft links "train-images", "train-labels", "test-images", and "test-labels" for these files.
3. Test it out (see code below).
4. Feel free to poke around with the algorithm constants to see if you can reduce the error rate.
5. If you're curious about the incorrect predictions, open up the viewer. (They're automatically saved to a file called `incorrect_predictions` and thereby automatically displayed.)

```python
import graddesc, softmax, mlp
train = graddesc.training_examples() # takes a few seconds
test = graddesc.testing_examples()
graddesc.alg = mlp # or softmax
graddesc.run(train, test, test) # takes quite a while
```

Note: currently the validation examples (the second parameter to `graddesc.run`) aren't used for anything.

## IDX Manipulation

`idx.py` contains a more general IDX reader/writer, as well as some helper methods for preprocessing the MNIST data (cropping the images, for example).

## Future work

* Compare our approach to Nielsen's http://neuralnetworksanddeeplearning.com/ (which achieves 2% error rate with only one hidden layer)
* Try using some other tricks: dropout, max-norm regularization, generative pre-training, other activation functions (ReLU, maxout)
* GPU optimization

