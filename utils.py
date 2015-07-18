import numpy

REGULARIZATION_PENALTY = 0.01

def regularization_gradient_ignoring_constant(two_dimensional_matrix):
  # We don't want to penalize the constant term for each parameter vector
  zeroer = numpy.identity(two_dimensional_matrix.shape[1])
  zeroer[0,0] = 0
  zeroed_matrix = numpy.dot(two_dimensional_matrix, zeroer)
  return numpy.dot(2 * REGULARIZATION_PENALTY, zeroed_matrix)

