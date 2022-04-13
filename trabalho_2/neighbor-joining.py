#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install pandas')


# In[24]:
from typing import Tuple, List, Union, Any

import pandas as pd
import numpy as np
import string

np.set_printoptions(precision=3)


# In[2]:


def load_distance_matrix(file_name: str) -> np.ndarray:
    sample = np.loadtxt(file_name)
    index = [chr(65 + i) for i in range(sample.shape[0])]
    sample = sample[1:, :-1]
    return pd.DataFrame(sample, index=index[1:], columns=index[:-1])


# In[3]:


def get_argmin_value(matrix: np.ndarray) -> tuple[int, int]:
    return tuple(np.argwhere(matrix == np.min(matrix))[0])


# In[4]:


def init_index_dict(len_: int) -> list[str]:
    index_dict = [chr(65 + i) for i in range(len_)]
    return index_dict


# In[14]:


def get_u_mean_distances(distance_matrix: np.ndarray) -> np.array:
    u_mean_distances = []
    for i in range(distance_matrix.shape[0]):
        u_mean_distances.append(distance_matrix.values[:, i].mean())
    u_mean_distances = np.array(u_mean_distances)
    return u_mean_distances


def get_pairs_distance(distance_matrix: np.ndarray, u_mean_distances: np.array) -> np.ndarray:
    pairs_distance = np.zeros(distance_matrix.shape)
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            pairs_distance[i, j] = distance_matrix.values[i, j] - u_mean_distances[i] - u_mean_distances[j]
    pairs_distance = pairs_distance.transpose()
    return pairs_distance


# In[15]:


def get_distance_from_smallest_pair_to_new_node(distance_matrix: np.ndarray,
                                                u_mean_distances: np.array,
                                                smallest_pair: tuple[int, int]) -> tuple[int, int]:
    d0 = distance_matrix.values[smallest_pair] / 2 + (
                u_mean_distances[smallest_pair[0]] - u_mean_distances[smallest_pair[1]]) / 2

    d1 = distance_matrix.values[smallest_pair] / 2 + (
                u_mean_distances[smallest_pair[1]] - u_mean_distances[smallest_pair[0]]) / 2
    return d0, d1


# In[16]:


def get_new_values(distance_matrix: np.ndarray,
                   smallest_pair: tuple[int, int],
                   pairs_distance: np.ndarray) -> tuple[list[float], list[float]]:

    new_row = []
    for i in range(distance_matrix.shape[1]):
        if i not in smallest_pair:
            new_value = (distance_matrix.values[i, smallest_pair[0]] + distance_matrix.values[i, smallest_pair[1]] -
                         pairs_distance[smallest_pair]) / 2
            new_row.append(new_value)

    new_column = []
    for i in range(distance_matrix.shape[0]):
        if i not in smallest_pair:
            new_value = (distance_matrix.values[i, smallest_pair[0]] + distance_matrix.values[i, smallest_pair[1]] -
                         pairs_distance[smallest_pair]) / 2
            new_column.append(new_value)

    return new_row, new_column


def get_new_distance_matrix(distance_matrix: np.ndarray,
                            smallest_pair: tuple[int, int],
                            new_row: list[int],
                            new_column: list[int],
                            iteration: int) -> np.ndarray:
    rows_columnns_to_remove = [distance_matrix.index[smallest_pair[0]],
                               distance_matrix.columns[smallest_pair[1]]]

    distance_matrix.drop(index=rows_columnns_to_remove,
                         errors="ignore",
                         inplace=True)

    distance_matrix.drop(columns=rows_columnns_to_remove,
                         errors="ignore",
                         inplace=True)

    distance_matrix[f"U{iteration}"] = new_column

    new_row.append(0)
    distance_matrix.loc[f"U{iteration}"] = new_row

    return distance_matrix


# In[115]:


def main():
    distance_matrix = load_distance_matrix("input.txt")
    nick_tree = []
    rows, columns = distance_matrix.shape

    for iteration in range(columns - 1):
        print(distance_matrix)
        u_mean_distances = get_u_mean_distances(distance_matrix)
        pairs_distance = get_pairs_distance(distance_matrix, u_mean_distances)
        smallest_pair = get_argmin_value(pairs_distance)
        d0, d1 = get_distance_from_smallest_pair_to_new_node(distance_matrix,
                                                             u_mean_distances,
                                                             smallest_pair)
        nick_tree.append((f"U{iteration}", distance_matrix.index[smallest_pair[0]], d0))
        nick_tree.append((f"U{iteration}", distance_matrix.columns[smallest_pair[1]], d1))
        new_row, new_column = get_new_values(distance_matrix,
                                    smallest_pair,
                                    pairs_distance)

        # break
        distance_matrix = get_new_distance_matrix(distance_matrix,
                                                  smallest_pair,
                                                  new_row,
                                                  new_column,
                                                  iteration)
        rows, columns = distance_matrix.shape
        print("*" * 80)
        #print("\n\n\n")
        # break

