"""GaussianMLPRegressorModel."""
import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel


class GaussianMLPRegressorModel(GaussianMLPModel):
    """
    GaussianMLPRegressor based on garage.tf.models.Model class.

    This class can be used to perform regression by fitting a Gaussian
    distribution to the outputs.

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity: Nonlinearity for each hidden layer in
            the std network.
        std_output_nonlinearity: Nonlinearity for output layer in
            the std network.
        std_parametrization (str): How the std should be parametrized. There
            are a few options:
        - exp: the logarithm of the std will be stored, and applied a
            exponential transformation
        - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='GaussianMLPRegressorModel',
                 **kwargs):
        super().__init__(output_dim=output_dim, name=name, **kwargs)
        self._input_shape = input_shape

    def network_output_spec(self):
        """Network output spec."""
        return [
            'sample', 'means', 'log_stds', 'std_param', 'normalized_means',
            'normalized_log_stds', 'x_mean', 'x_std', 'y_mean', 'y_std', 'dist'
        ]

    def _build(self, state_input, name=None):
        with tf.compat.v1.variable_scope('normalized_vars'):
            x_mean_var = tf.compat.v1.get_variable(
                name='x_mean',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            x_std_var = tf.compat.v1.get_variable(
                name='x_std_var',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)
            y_mean_var = tf.compat.v1.get_variable(
                name='y_mean_var',
                shape=(1, self._output_dim),
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            y_std_var = tf.compat.v1.get_variable(
                name='y_std_var',
                shape=(1, self._output_dim),
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)

        normalized_xs_var = (state_input - x_mean_var) / x_std_var

        sample, normalized_mean, normalized_log_std, std_param, dist = super(
        )._build(normalized_xs_var)

        with tf.name_scope('mean_network'):
            means_var = normalized_mean * y_std_var + y_mean_var

        with tf.name_scope('std_network'):
            log_stds_var = normalized_log_std + tf.math.log(y_std_var)

        return (sample, means_var, log_stds_var, std_param, normalized_mean,
                normalized_log_std, x_mean_var, x_std_var, y_mean_var,
                y_std_var, dist)
