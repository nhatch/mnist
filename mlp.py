import numpy
import math
import utils

NUM_CLASSES = 10
NUM_HIDDEN_UNITS = 64
# UNITS_LAYERS -- the number of units in each layer, not including the input layer,
#                 in order of increasing abstractness (i.e. output layer at the end)
UNITS_LAYERS = [NUM_HIDDEN_UNITS, NUM_CLASSES]
# ACTIVATION_FUNCTION -- the output of a hidden unit as a function of the weighted
#                        sum of its inputs
ACTIVATION_FUNCTION = math.tanh
DERIVATIVE_OF_ACTIVATION_FUNCTION = lambda(x): 1 - math.tanh(x)**2

def initial_parameters(num_input_units):
  lower_layer_sizes = [num_input_units] + UNITS_LAYERS[:-1]
  upper_layer_sizes = UNITS_LAYERS
  dimension_pairs = zip(upper_layer_sizes, lower_layer_sizes)
  parameter_layers = map(lambda(dimension_pair): _random_weights(*dimension_pair), dimension_pairs)
  # For convenience, even though these layers have different dimensions,
  # we wrap them in a numpy.array. This lets us add, subtract, and multiply
  # by scalars without explicitly iterating through the array.
  return numpy.array(parameter_layers)

def _random_weights(num_upper_units, num_lower_units):
  sigma = 1 / math.sqrt(num_upper_units)
  # Initialize weights and biases by drawing from N(0, sigma**2).
  # This helps avoid saturation: http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
  # The +1 adds an extra column for the bias terms.
  return numpy.dot(sigma, numpy.random.randn(num_upper_units, num_lower_units+1))

def regularization_gradient(parameters):
  return numpy.array(map(utils.regularization_gradient_ignoring_bias, parameters))

def loss_gradient_for_example(parameters, example):
  datum = example[0]
  label = example[1]
  unit_activations, derivative_unit_activations = _calculate_unit_activations(parameters, datum)
  # A "unit input partial" is the partial derivative of the loss function with
  # respect to the input to a particular unit.
  # The "input" to a unit is the weighted sum of its inputs from the previous
  # layer of the neural net, including bias.
  # For our loss function, and because the activation function for output units
  # is the identity, the unit input partial for an output unit is simply the
  # difference between the unit's activation (= the unit's input) and the desired
  # activation for the unit (1 for the correct label, 0 otherwise).
  output_unit_input_partials = unit_activations[-1]
  output_unit_input_partials[label] -= 1
  # The unit input partials for hidden units are calculated using backpropagation.
  unit_input_partials = _calculate_unit_input_partials(parameters, derivative_unit_activations, output_unit_input_partials)
  return _calculate_gradient(unit_input_partials, unit_activations)

def _calculate_unit_activations(parameters, datum, calculate_derivatives=True):
  unit_activations = []
  if calculate_derivatives:
    derivative_unit_activations = []
  previous_layer_outputs = datum
  for index, parameter_layer in enumerate(parameters):
    previous_layer_outputs = numpy.insert(previous_layer_outputs, 0, 1) # for the bias
    unit_activations.append(previous_layer_outputs)
    weighted_sums = numpy.dot(parameter_layer, previous_layer_outputs)
    if index < len(parameters) - 1:
      previous_layer_outputs = map(ACTIVATION_FUNCTION, weighted_sums)
      # For backpropagation, we need to know the derivative of the activation of
      # each hidden unit with respect to the input to that unit.
      if calculate_derivatives:
        derivatives = map(DERIVATIVE_OF_ACTIVATION_FUNCTION, weighted_sums)
        derivative_unit_activations.append(derivatives)
    else:
      # For the output units, the activation function is the identity.
      previous_layer_outputs = weighted_sums
      unit_activations.append(previous_layer_outputs)
  if calculate_derivatives:
    return unit_activations, derivative_unit_activations
  else:
    return unit_activations

def _calculate_unit_input_partials(parameters, derivative_unit_activations, output_unit_input_partials):
  unit_input_partials = [output_unit_input_partials]
  previous_unit_input_partials_layer = output_unit_input_partials
  params_with_upper_deltas = zip(parameters[1:], derivative_unit_activations)
  for parameter_layer, derivative_layer in reversed(params_with_upper_deltas):
    # We can calculate that the unit input partial for a hidden unit T is
    #   h'(a_T) * \sum_{S:T->S} w_TS \delta_S
    # where a_T is the input to T, h' is the derivative of the activation
    # funcion for T, T->S means that T sends its output to S, w_TS is the
    # weight that S gives to T's output, and \delta_S is the unit input
    # partial for S.
    # The following line calculates the right-hand side of that product
    # for all hidden units in the current layer.
    sums = numpy.dot(previous_unit_input_partials_layer, parameter_layer)
    # We ignore the constant, since it has no input (hence no unit input partial).
    sums = sums[1:]
    # We multiply each sum by the appropriate derivative.
    previous_unit_input_partials_layer = numpy.multiply(sums, derivative_layer)
    unit_input_partials.insert(0, previous_unit_input_partials_layer)
  return unit_input_partials

def _calculate_gradient(unit_input_partials, unit_activations):
  # The partial derivative of the loss function with respect to a particular
  # weight in the parameter array is the product of the unit input partial
  # for the "target unit" of that weight and the partial derivative of the
  # input to the target unit with respect to that weight. The latter partial
  # is just the activation of the "source unit". So, to calculate the
  # gradient, we just need to multiply together the correct unit activations
  # and unit input partials.
  layers = zip(unit_input_partials, unit_activations[:-1])
  layer_gradients = map(lambda(layer): _gradient_for_layer(*layer), layers)
  return numpy.array(layer_gradients)

def _gradient_for_layer(target_unit_input_partials, source_unit_activations):
  sua_reshape = source_unit_activations.reshape(1, len(source_unit_activations))
  tuip_reshape = target_unit_input_partials.reshape(len(target_unit_input_partials), 1)
  return numpy.dot(tuip_reshape, sua_reshape)

def norm(parameter_type_array):
  # Since each parameter layer has different dimensions, we have to manually
  # add the norm-squareds and take the square root.
  norm_squareds = map(lambda(x): numpy.linalg.norm(x)**2, parameter_type_array)
  return math.sqrt(sum(norm_squareds))

def classification_loss_for_example(parameters, example):
  datum = example[0]
  label = example[1]
  expected_outputs = numpy.zeros(NUM_CLASSES); expected_outputs[label] = 1
  actual_outputs = _calculate_unit_activations(parameters, datum, False)[-1]
  # Standard squared loss, divided by two to simplify gradient calculation
  return 0.5 * numpy.linalg.norm(numpy.subtract(expected_outputs, actual_outputs))**2

def regularization_loss_for_parameters(parameters):
  return sum(map(utils.regularization_loss_ignoring_bias, parameters))

def predict(parameters, datum):
  outputs = _calculate_unit_activations(parameters, datum, False)[-1]
  return numpy.argmax(outputs)

def save_parameters(parameters):
  global mlp_parameters
  mlp_parameters = parameters
  import idx
  idx.write("parameters", utils.square_parameters(parameters[0]))

def numerical_approximation_of_gradient(parameters, example):
  delta = 2**-13
  base_loss = _total_loss_for_example(parameters, example)
  gradient_layers = []
  for parameter_layer in parameters:
    gradient_layer = []
    for i in range(len(parameter_layer)):
      row = parameter_layer[i]
      gradient_row = []
      for j in range(len(row)):
        row[j] += delta
        new_loss = _total_loss_for_example(parameters, example)
        gradient_row.append((new_loss - base_loss) / delta)
        row[j] -= delta
      gradient_layer.append(gradient_row)
    gradient_layers.append(numpy.array(gradient_layer))
  return numpy.array(gradient_layers)

def _total_loss_for_example(parameters, example):
  return classification_loss_for_example(parameters, example) + regularization_loss_for_parameters(parameters)

