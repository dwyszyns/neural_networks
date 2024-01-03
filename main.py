from math import cos, sin, sqrt
from neural_network import GradientNetwork, EvolutionNetwork, arctan, d_arctan
import numpy as np
import matplotlib.pyplot as plt


def f(x) -> float:
    return x**2 * sin(x) + 100 * sin(x) * cos(x)


def generate_training_data(func, n=100, bounds=(-40, 40)):
    x = np.linspace(bounds[0], bounds[1], n)
    y = np.array([func(temp) for temp in x])
    return x, y


def make_plot(network, bounds, func):
    x = np.linspace(bounds[0], bounds[1], 1000)
    y = [func(xi) for xi in x]
    plt.plot(x, y)

    x_predicted = generate_training_data(func, 100, bounds)[0]
    y_predicted = [network.feed_forward(xi)[2] for xi in x_predicted]
    for xi, yi in zip(x_predicted, y_predicted):
        plt.scatter(xi, yi)

    plt.title("Percepton wielowarstwowy - 2 warstwy ukryte, po 10 neuronow")
    plt.savefig("evol-lay-2-neuron-10.png")
    plt.show()


class params_t:
    def __init__(self, hidden_neurons=[10, 10], bounds=(-10, 10), func=f):
        self.hidden_neurons = hidden_neurons
        self.bounds = bounds
        self.func = func


class network_params_t:
    def __init__(
        self,
        num_of_samples=100,
        iterations=1000,
        batch_size=20,
        activation_func=arctan,
        d_activation_func=d_arctan,
    ):
        self.num_of_samples = num_of_samples
        self.iterations = iterations
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func


if __name__ == "__main__":
    params = params_t([10, 10], (-10, 10), f)
    network_params = network_params_t(100, 200, 20, arctan, d_arctan)

    x, y = generate_training_data(
        params.func, network_params.num_of_samples, params.bounds
    )

    network = EvolutionNetwork(
        params.hidden_neurons,
        1,
        network_params.activation_func,
        network_params.d_activation_func,
        population_size=100,
    )
    network.train(x, y, network_params.iterations)
    make_plot(network, params.bounds, params.func)

    network = GradientNetwork(
        params.hidden_neurons,
        1,
        network_params.activation_func,
        network_params.d_activation_func,
        beta=0.001,
    )
    network.train(x, y, network_params.iterations, network_params.batch_size)
    make_plot(network, params.bounds, params.func)
