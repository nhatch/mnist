import numpy
import math

REGULARIZATION_PENALTY = 0.01

def regularization_loss_ignoring_bias(two_dimensional_matrix):
  matrix_without_bias = _remove_bias(two_dimensional_matrix)
  # Standard L2-norm regularization
  return REGULARIZATION_PENALTY * numpy.linalg.norm(matrix_without_bias) ** 2

def regularization_gradient_ignoring_bias(two_dimensional_matrix):
  matrix_without_bias = _remove_bias(two_dimensional_matrix)
  return numpy.dot(2 * REGULARIZATION_PENALTY, matrix_without_bias)

def _remove_bias(two_dimensional_matrix):
  copied_matrix = numpy.copy(two_dimensional_matrix)
  # We don't want to penalize the bias term for each parameter vector
  for row in copied_matrix:
    row[0] = 0
  return copied_matrix

def square_parameters(parameters):
  side_length = int(math.sqrt(len(parameters[0]) - 1))
  to_square = lambda(layer): layer[1:].reshape(side_length, side_length)
  return map(to_square, _normalize(parameters))

def _normalize(parameters):
  without_bias = _remove_bias(parameters)
  factor = 255 / (without_bias.max() - without_bias.min())
  shifted = numpy.add(without_bias, -without_bias.min())
  return numpy.dot(shifted, factor).astype(int)

