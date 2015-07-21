import numpy
import math
import utils

NUM_CLASSES = 10

def initial_parameters(data_dimension):
  return numpy.zeros((NUM_CLASSES, 1 + data_dimension))

def regularization_gradient(parameters):
  return utils.regularization_gradient_ignoring_constant(parameters)

def probability_gradient_for_example(parameters, example):
  datum = numpy.insert(example[0], 0, 1)
  label = example[1]
  probabilities = map(lambda(lbl): _probability_of_example(parameters, datum, lbl), range(NUM_CLASSES))
  scalars = map(lambda(lbl): [probabilities[lbl] - _delta(lbl, label)], range(NUM_CLASSES))
  return numpy.dot(scalars, datum.reshape((1, len(datum))))

def _probability_of_example(parameters, datum, lbl):
  dot_products = numpy.dot(parameters, datum)
  exponentiated_dot_products = map(lambda(x): math.exp(x), dot_products)
  return exponentiated_dot_products[lbl] / sum(exponentiated_dot_products)

def _delta(x, y):
  return 1 if x == y else 0

def predict(parameters, datum):
  datum = numpy.insert(datum, 0, 1)
  exponents = numpy.dot(parameters, datum)
  return numpy.argmax(exponents)

def distance_between(a, b):
  difference = numpy.subtract(a, b)
  return numpy.linalg.norm(difference)

def save_parameters(parameters):
  import idx
  idx.write("parameters", utils.square_parameters(parameters))
