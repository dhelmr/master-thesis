import math
from typing import Optional

from tqdm import tqdm

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class KCenter(BuildingBlock):
    def __init__(self,
                 feature: BuildingBlock, k: int):
        """
        Building Block that implements the k-center anomaly decider approach

        Idea: find k centers in validation data with greedy algorithm
        find the maximum radius r of datapoints in validation data to that centers

        if the euclidian distance of a new datapoint is > r the datapoint is considered an anomaly
        if the euclidian distance is < r the datapoint is considered benign
        @param feature: the input feature building block
        @param k: the number of centers to be found in the validation data
        """
        super().__init__()
        self._feature = feature
        self._dependency_list = []
        self._dependency_list.append(self._feature)
        self._k = k

        self._datapoints = []
        self._distance_matrix = []

        self._centers = []
        self._center_indices = []
        self._max_radius = 0.0

        self._cache = {}

    def depends_on(self):
        return self._dependency_list

    def is_decider(self):
        return True

    def val_on(self, syscall: Syscall):
        """
        adds validation datapoints to distinct datapoint list
        @param syscall: the current validation system call
        """
        feature_input = self._feature.get_result(syscall)

        # cast int or float to list
        if type(feature_input) == int or type(feature_input) == float:
            feature_input = [feature_input]

        if feature_input is not None:
            if list(feature_input) not in self._datapoints:
                self._datapoints.append(list(feature_input))

    def fit(self):
        """
        calculates the distance matrix for all datapoints
        calls method to find centers in data
        calls method to determine the maximum radius r
        """
        for point_a in tqdm(self._datapoints, desc="Calculating distance matrix".rjust(27)):
            distances = []
            for point_b in self._datapoints:
                # calculate euclidian distance
                distance = math.dist(point_a, point_b)
                distances.append(distance)
            self._distance_matrix.append(distances)
        self._find_k_centers()
        self._find_max_radius()

    def _calculate(self, syscall: Syscall) -> Optional[bool]:
        """
        finds the nearest center for new datapoint
        checks if its euclidian distance is > r and decides if datapoint is anomal or not
        @param syscall:
        @return:
        """
        feature_input = self._feature.get_result(syscall)

        # cast int to list
        if type(feature_input) == int:
            feature_input = [feature_input]

        if feature_input is not None:
            # caching the result
            if tuple(feature_input) in self._cache.keys():
                return self._cache[tuple(feature_input)]
            else:
                # find the nearest center
                min_distance = 10 ** 9
                for center in self._centers:
                    current_distance = math.dist(list(feature_input), center)
                    if current_distance < min_distance:
                        min_distance = current_distance

                if min_distance > self._max_radius:
                    self._cache[tuple(feature_input)] = True
                    return True
                else:
                    self._cache[tuple(feature_input)] = False
                    return False
        else:
            return None

    def _find_k_centers(self):
        """
        greedy algorithm that finds the k centers in datapoints
        """
        n = len(self._distance_matrix)
        # filling the distance list with big values
        dist = [10**9] * n

        max_index = 0
        for _ in tqdm(range(self._k), desc="Calculating centers".rjust(27)):
            # add point to center indices list and center list
            self._center_indices.append(max_index)
            self._centers.append(self._datapoints[max_index])
            for j in range(n):
                # updating the distance
                # of the center to their
                # closest centers
                dist[j] = min(dist[j], self._distance_matrix[max_index][j])

            # updating the index of the
            # center with the maximum
            # distance to it's closest center
            max_index = self._calc_max_index(dist, n)

    @staticmethod
    def _calc_max_index(dist, n):
        """
        helper method to find index of max far datapoint
        @param dist: distance between datapoints
        @param n: current counter
        @return: the max index
        """
        max_index = 0
        for i in range(n):
            if dist[i] > dist[max_index]:
                max_index = i
        return max_index

    def _find_max_radius(self):
        """
        finds the maximum radius over all datapoints to their nearest centers
        """
        for point_index, point in enumerate(tqdm(self._datapoints, desc="Calculating radius".rjust(27))):
            nearest_center_distance = 10 ** 9
            # finding the nearest center for datapoint by lookup in distance matrix
            for center_index in self._center_indices:
                current_distance = self._distance_matrix[point_index][center_index]
                if current_distance < nearest_center_distance:
                    nearest_center_distance = current_distance

            # the maximum of the minimum distance for all centers over all points is the max radius r
            if nearest_center_distance > self._max_radius:
                self._max_radius = nearest_center_distance
