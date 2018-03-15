'''Export single neuron neural network model for TensorFlow Serving'''
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer('model_version', 1, 'Model version number')
tf.app.flags.DEFINE_string('export_dir', '/tmp/nn', 'Export model directory')
FLAGS = tf.app.flags.FLAGS

# Set up sample points perturbed away from the ideal linear relationship
# y = 0.5*x + 2.5
num_examples = 60
points = np.array([np.linspace(-1, 5, num_examples),
                   np.linspace(2, 5, num_examples)])
points += np.random.randn(2, num_examples)
train_x, train_y = points
# Include a 1 to use as the bias input for neurons
train_x_with_bias = np.array([(1., d) for d in train_x]).astype(np.float32)

# Training parameters
training_steps = 100
learning_rate = 0.001

with tf.Session() as sess:
  # Set up all the tensors, variables, and operations.
  input = tf.constant(train_x_with_bias)
  target = tf.constant(np.transpose([train_y]).astype(np.float32))
  # Set up placeholder for making model predictions (separate from training)
  x = tf.placeholder('float', shape=[None, 1])
  # Initialize weights with small random values
  weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

  tf.global_variables_initializer().run()

  # Calculate the current prediction error
  train_y_predicted = tf.matmul(input, weights)
  train_y_error = tf.subtract(train_y_predicted, target)

  # Define prediction operation
  y = tf.matmul(x, weights[1:]) + weights[0]

  # Compute the L2 loss function of the error
  loss = tf.nn.l2_loss(train_y_error)

  # Train the network using an optimizer that minimizes the loss function
  update_weights = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss)

  for _ in range(training_steps):
    # Repeatedly run the operations, updating the TensorFlow variable.
    update_weights.run()

  ## Export the Model

  # Create a SavedModelBuilder
  export_path_base = FLAGS.export_dir
  export_path = os.path.join(export_path_base, str(FLAGS.model_version))
  print('Exporting trained model to', export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  # Build signature inputs and outputs
  tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_output = tf.saved_model.utils.build_tensor_info(y)

  # Create the prediction signature
  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'input': tensor_info_input},
      outputs={'output': tensor_info_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  # Provide legacy initialization op for compatibility with older version of tf
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

  # Build the model
  builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      'prediction':
      prediction_signature,
    },
  legacy_init_op=legacy_init_op)

  # Save the model
  builder.save()

print('Complete')
