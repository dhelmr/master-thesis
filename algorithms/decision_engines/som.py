import math
import random
import sys

from matplotlib import pyplot as plt

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from minisom import MiniSom
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm, linalg

ACTIVATION_DISTANCE_EXP = "exponential"
ACTIVATION_FREQ_WEIGHTED = "freq-eucl"


class Som(BuildingBlock):
    def __init__(self, input_vector: BuildingBlock, epochs: int = 50, sigma: float = 1.0, learning_rate: float = 0.5,
                 max_size: int = None, activation_distance: str = None, exp_euclidean_beta=3, size=None, size_factor=1,
                 sampling_prob_alpha=0):
        """
            Anomaly Detection Engine based on Teuvo Kohonen's Self-Organizing-Map (SOM)

            Uses the MiniSOM Implementation from https://github.com/JustGlowing/minisom

            Vector size is derived from input data size

            Parameters:

                epochs: Number of Epochs for training (Iterations over complete training data)
                sigma: Spread of the neighborhood function, needs to be adequate to the dimensions of the map
                    (at iteration t: have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
                learning_rate: Initial learning rate
                    (at iteration t: learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
                max_size: the maximum size for size estimation
                size: set if size shall not be estimated dynamically (SOM is always initialized quadratic)
        """
        super().__init__()
        self._input_vector = input_vector
        self._dependency_list = [input_vector]
        self._sigma = sigma
        self._learning_rate = learning_rate
        self._buffer = dict()
        self._epochs = epochs
        self._som = None
        self._cache = {}
        self._activation_distance = activation_distance
        self._exp_euclidean_beta = exp_euclidean_beta
        self._max_size = max_size
        self._size = size
        self._size_factor = size_factor
        self._sampling_prob_alpha = sampling_prob_alpha
        if sampling_prob_alpha < 0 or sampling_prob_alpha >= 1:
            raise ValueError("sampling_prob_alpha is %s but must be in [0,1) interval!" % sampling_prob_alpha)
        self.custom_fields = {}

    def depends_on(self):
        return self._dependency_list

    def _get_or_estimate_som_size(self):
        """
            Estimates the SOM size by adding 1 to the root of number of vectors in training data.
            As training data is distinct this leads to a slightly higher number of Neurons than distinct input vectors

            Idea: Have at least one Neuron for every single distinct input vector

            if a fixed size is given on initialization of the som the estimation will be skipped
        """
        if self._size is None:
            som_size = round(math.sqrt(
                len(self._buffer) * self._size_factor
            ), 0)

            som_size += 1
            if self._max_size is not None and som_size > self._max_size:
                return self._max_size
            else:
                return int(som_size)
        else:
            return self._size

    def train_on(self, syscall: Syscall):
        """
            creates distinct input data buffer used for training
        """
        input_vector = self._input_vector.get_result(syscall)
        if input_vector is not None:
            if input_vector not in self._buffer:
                self._buffer[input_vector] = 0
            self._buffer[input_vector] += 1

    def fit(self):
        """
            finalizes the training step for the som
        """
        print(f"som.train_set: {len(self._buffer)} ".rjust(27))
        # print(self._buffer)
        som_size = self._get_or_estimate_som_size()
        print("Buffer Size:", len(self._buffer), "Som Size", som_size)
        # vector_size = len(self._buffer[0])
        vector_size = len(next(iter(self._buffer)))

        kwargs = {
            "random_seed": 1,
            "sigma": self._sigma,
            "learning_rate": self._learning_rate,
        }
        if self._activation_distance is not None:
            kwargs["activation_distance"] = self._activation_distance
            if self._activation_distance == ACTIVATION_DISTANCE_EXP:
                kwargs["activation_distance"] = self._exp_distance
            elif self._activation_distance == ACTIVATION_FREQ_WEIGHTED:
                kwargs["activation_distance"] = self._freq_weighted_distance
        self._som = MiniSom(som_size, som_size, vector_size, **kwargs)

        for epoch in tqdm(range(self._epochs), desc='Training SOM'.rjust(27)):
            for vector, freq in self._buffer.items():
                if self._sample_vector(freq):
                    self._som.update(vector, self._som.winner(vector), epoch, self._epochs)

    def _calculate(self, syscall: Syscall):
        """
            calculates euclidean distance between input and codebook vector which is used as anomaly score

            Returns:
                distance (float): euclidian distance/anomaly score
        """
        input_vector = self._input_vector.get_result(syscall)
        if input_vector is not None:
            if input_vector not in self._cache:
                codebook_vector = np.array(self._som.quantization([input_vector])[0])
                vector = np.array(input_vector)
                # distance = norm(vector - codebook_vector)
                distance = self._som._activation_distance(vector, codebook_vector)
                self._cache[input_vector] = distance
            else:
                distance = self._cache[input_vector]
            return distance
        else:
            return None

    def show_distance_plot(self):
        """
            Creates distance plot for trained SOM, ach cell is the normalised sum of the distances
            between a neuron and its neighbours.
        """
        plt.figure()
        plt.pcolor(self._som.distance_map(), cmap='gnuplot')
        plt.colorbar()

        plt.show()

    def calculate_errors(self):
        """
            Calculates Errors and adds them to public custom field dict.

            Errors are only calculated for training data.

            Quantization Error:
                average distance between each input sample and its best matching neuron

            Topographic Error:
                Is computed by finding the best-matching and second-best-matching neuron
                in the map for each input and then evaluating the positions.

                A sample for which these two nodes are not adjacent counts as
                an error. The topographic error is given by the
                the total number of errors divided by the total of samples.

                If the topographic error is 0, no error occurred.
                If 1, the topology was not preserved for any of the samples.
        """
        self.custom_fields['training_quantization_error'] = self._som.quantization_error(list(self._buffer))
        self.custom_fields['training_topographicn_error'] = self._som.topographic_error(list(self._buffer))

    def _exp_distance(self, x, w):
        return exp_euclidean(x, w, self._exp_euclidean_beta)

    def _freq_weighted_distance(self, x, w):
        if len(w.shape) == 3:  # for training (w is matrix of neurons)
            w = w[:, :, :-1]
        elif len(w.shape) == 1:  # for inference (w is winning neuron)
            w = w[:-1]
        else:
            raise ValueError("Unexpected shape of w: %s", w.shape)
        factor = x[-1:]
        dist = linalg.norm(x[:-1] - w, axis=-1) * factor
        if len(w.shape) == 1:
            return dist[0]
        else:
            return dist

    def _sample_vector(self, frequency) -> bool:
        probability = 1-(math.pow(self._sampling_prob_alpha, frequency))
        return random.random() < probability

def exp_euclidean(x, w, beta):
    return np.sqrt(1 - np.exp(-beta * linalg.norm(x - w, axis=-1)))
