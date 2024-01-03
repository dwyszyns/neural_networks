import numpy as np
from math import exp, atan
import cma


def arctan(z):
    return atan(z)


def sigmoid(z):
    return 1 / (1 + exp(-z))


def d_sigmoid(z):
    return exp(-z) / ((1 + exp(-z)) ** 2)


def d_arctan(z):
    return 1 / (1 + (z**2))


class NeuralNetwork:
    def __init__(
        self, hidden_layers, input_size, activation_func, d_activation_func
    ) -> None:
        self.hidden_layers = hidden_layers
        self.num_of_hidden_layers = len(hidden_layers)
        self.hidden_weights = []
        self.exit_weights = np.zeros(self.hidden_layers[-1] + 1)
        self.input_size = input_size
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.init_weights()

    def init_weights(self):
        rng = np.random.default_rng(0)
        bound = 1 / np.sqrt(self.input_size)
        for i in range(self.num_of_hidden_layers):
            uniform_len = self.input_size if i == 0 else self.hidden_layers[i - 1]
            self.hidden_weights.append(
                rng.uniform(-bound, bound, (self.hidden_layers[i], uniform_len + 1))
            )

    def loss(self, y, y_predicted):
        return np.square(y - y_predicted)

    def d_loss(self, y, y_predicted):
        return 2 * (y - y_predicted)

    def set_weights(self, flattened_params):
        offset = 0
        for i in range(self.num_of_hidden_layers):
            layer_size = self.hidden_layers[i]
            prev_layer_size = self.input_size if i == 0 else self.hidden_layers[i - 1]
            self.hidden_weights[i] = flattened_params[
                offset : offset + layer_size * (prev_layer_size + 1)
            ].reshape(layer_size, prev_layer_size + 1)
            offset += layer_size * (prev_layer_size + 1)
        self.exit_weights = flattened_params[offset:].reshape(
            self.hidden_layers[-1] + 1
        )

    def get_weights(self):
        return np.concatenate(
            [layer.flatten() for layer in self.hidden_weights + [self.exit_weights]]
        )

    def update_input_size(self, new_input_size):
        self.input_size = new_input_size
        self.init_weights()

    def update_hidden_layers(self, new_hidden_layers):
        self.hidden_layers = new_hidden_layers
        self.num_of_hidden_layers = len(new_hidden_layers)
        self.init_weights()

    def update_all_weights(self, flattened_params, new_input_size, new_hidden_layers):
        self.update_input_size(new_input_size)
        self.update_hidden_layers(new_hidden_layers)
        self.set_weights(flattened_params)

    def generate_batches(self, x, y, batch_size):
        data = np.c_[x, y]
        np.random.shuffle(data)
        num_of_batches = -(-len(data) // batch_size)
        return np.array_split(data, num_of_batches)

    def calculate_layer_output(self, x, weights):
        x = np.append(x, 1)
        layer_output = np.dot(x, weights.T)
        return x, layer_output

    def feed_forward(self, x):
        hidden_layers_outputs = []
        hidden_layers_outputs_act = []
        for i in range(self.num_of_hidden_layers):
            x, layer_output = self.calculate_layer_output(x, self.hidden_weights[i])
            layer_output_act = np.array([self.activation_func(v) for v in layer_output])
            hidden_layers_outputs.append(layer_output)
            hidden_layers_outputs_act.append(layer_output_act)
            x = layer_output_act
        _, exit_layer_output = self.calculate_layer_output(x, self.exit_weights)
        return hidden_layers_outputs, hidden_layers_outputs_act, exit_layer_output


class GradientNetwork(NeuralNetwork):
    def __init__(
        self, hidden_layers, input_size, activation_func, d_activation_func, beta=0.01
    ):
        super().__init__(hidden_layers, input_size, activation_func, d_activation_func)
        self.beta = beta

    def calc_sum_deriv(self, current_layer, current_neuron, y):
        if current_layer == self.num_of_hidden_layers - 1:
            return (
                self.d_loss(self.exit_layer_output, y)
                * self.exit_weights[current_neuron]
            )
        y_predicted = 0
        for i in range(self.hidden_layers[current_layer + 1]):
            y_predicted += (
                self.calc_sum_deriv(current_layer + 1, i, y)
                * self.hidden_weights[current_layer + 1][i][current_neuron]
            )
        return y_predicted * self.d_activation_func(
            self.hidden_sums[current_layer][current_neuron]
        )

    def calc_gradient(self, y, x):
        exit_gradient = np.zeros(self.exit_weights.shape)
        for exit_neuron in range(len(self.exit_weights)):
            exit_gradient[exit_neuron] = self.d_loss(self.exit_layer_output, y)
            if exit_neuron != len(self.exit_weights) - 1:
                exit_gradient[exit_neuron] *= self.hidden_outputs[-1][exit_neuron]
        hidden_gradients = []
        for hidden_layer in range(self.num_of_hidden_layers):
            hidden_gradient = np.zeros((self.hidden_weights[hidden_layer].shape))
            for current_neuron in range(hidden_gradient.shape[0]):
                for prev_layer_neuron in range(hidden_gradient.shape[1]):
                    activation_factor = self.d_activation_func(
                        self.hidden_sums[hidden_layer][current_neuron]
                    )
                    if hidden_layer == 0:
                        activation_factor *= x[prev_layer_neuron]
                    else:
                        if prev_layer_neuron < hidden_gradient.shape[1] - 1:
                            activation_factor *= self.hidden_outputs[hidden_layer - 1][
                                prev_layer_neuron
                            ]
                    if hidden_layer == self.num_of_hidden_layers - 1:
                        loss_factor = (
                            self.d_loss(self.exit_layer_output, y)
                            * self.exit_weights[current_neuron]
                        )
                    else:
                        loss_factor = 0
                        for z in range(self.hidden_layers[hidden_layer + 1]):
                            loss_factor += (
                                self.calc_sum_deriv(hidden_layer + 1, z, y)
                                * self.hidden_weights[hidden_layer + 1][z][
                                    current_neuron
                                ]
                            )
                    hidden_gradient[current_neuron][prev_layer_neuron] = (
                        loss_factor * activation_factor
                    )
            hidden_gradients.append(hidden_gradient)
        return hidden_gradients, exit_gradient

    def train(self, X, Y, iterations=10, batch_size=50):
        for _ in range(iterations):
            batches = self.generate_batches(X, Y, batch_size)
            for batch in batches:
                batch_hidden_grad_sum = []
                for hidden_layer in range(self.num_of_hidden_layers):
                    batch_hidden_grad_sum.append(
                        np.zeros((self.hidden_weights[hidden_layer].shape))
                    )
                batch_exit_grad_sum = np.zeros(self.exit_weights.shape)
                for x, y in batch:
                    (
                        self.hidden_sums,
                        self.hidden_outputs,
                        self.exit_layer_output,
                    ) = self.feed_forward(x)
                    x = np.append(x, 1)

                    (
                        batch_hidden_grad,
                        batch_exit_grad,
                    ) = self.calc_gradient(y, x)
                    batch_exit_grad_sum += batch_exit_grad / batch_size

                    for current_layer in range(self.num_of_hidden_layers):
                        batch_hidden_grad_sum[current_layer] += (
                            batch_hidden_grad[current_layer] / batch_size
                        )

            for current_layer in range(self.num_of_hidden_layers):
                self.hidden_weights[current_layer] -= self.beta * batch_hidden_grad_sum[current_layer]
            self.exit_weights -= self.beta * batch_exit_grad_sum


class EvolutionNetwork(NeuralNetwork):
    def __init__(
        self,
        hidden_layers,
        input_size,
        activation_func,
        d_activation_func,
        sigma=0.1,
        population_size=50,
    ):
        super().__init__(hidden_layers, input_size, activation_func, d_activation_func)
        self.sigma = sigma
        self.population_size = population_size

    def train(self, x, y, iterations=100):
        def cost_func(params):
            self.set_weights(params)
            y_pred = [self.feed_forward(xi)[2] for xi in x]
            loss = np.sum(np.square(y - y_pred))
            return loss

        params_init = np.concatenate(
            [layer.flatten() for layer in self.hidden_weights + [self.exit_weights]]
        )
        evolution_strategy = cma.CMAEvolutionStrategy(
            params_init,
            self.sigma,
            {"popsize": self.population_size, "maxiter": iterations},
        )
        evolution_strategy.optimize(cost_func)
        self.set_weights(evolution_strategy.result.xbest)
