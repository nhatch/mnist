import numpy
import math

REGULARIZATION_PENALTY = 0.004

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
  return map(lambda(v): _vector_to_ubyte_square(v, side_length), parameters)

def _vector_to_ubyte_square(parameter_vector, side_length):
  parameter_vector = parameter_vector[1:] # ignore the constant term
  factor = 255 / (parameter_vector.max() - parameter_vector.min())
  shifted_vector = numpy.add(parameter_vector, -parameter_vector.min())
  normalized_vector = numpy.dot(shifted_vector, factor).astype(int)
  return normalized_vector.reshape(side_length, side_length)

