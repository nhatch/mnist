# MNIST

An investigation into neural networks using the [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits.

Currently the softmax classifier has an error rate of about 10%.

## MNIST Viewer

A simple page for viewing MNIST images.

1. Download/unzip MNIST's "t10k-images-idx3-ubyte" into this directory.
2. Create a soft link "test-images" for this file: `ln -s t10k-images-idx3-ubyte test-images`
3. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
4. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

## Softmax Algorithm

An implementation of the softmax algorithm that can be used to classify MNIST images.

1. Download/unzip the MNIST data (four files total) into this directory.
2. Create soft links "train-images", "train-labels", "test-images", and "test-labels" for these files.
3. Test it out (see code below).
4. Feel free to poke around with the algorithm constants to see if you can reduce the error rate.
5. If you're curious about the incorrect predictions, open up the viewer. (They're automatically saved to a file called `incorrect_predictions` and thereby automatically displayed.)

```python
import softmax
train = softmax.training_examples() # takes a few seconds
test = softmax.testing_examples()
# train = train[:1000] # if you want it to finish faster
softmax.run(train, test) # takes quite a while
```

## IDX Manipulation

`idx.py` contains a more general IDX reader/writer, as well as some helper methods for preprocessing the MNIST data (cropping the images, for example).

