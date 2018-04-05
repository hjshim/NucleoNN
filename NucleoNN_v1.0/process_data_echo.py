from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FILE_NAME = 'echovirus_Cl_1per_NE.zip'

# Step 1: read data in a list of strings
def read_data(file_path): 
    """ Extract the first file enclosed in a zip file as a list of strings. """
    with zipfile.ZipFile(file_path) as f:
        alleles = tf.compat.as_str(f.read(f.namelist()[0])).split(",") # make sure no "," at the end
    alleles = list(map(int, alleles))
    return alleles

# Step 2: generate samples with allele frequencies
def generate_sample(index_alleles, context_window_size): 
    """ Process raw data into training pairs (according to the skip-gram model). """
    for index, center in enumerate(index_alleles):
        context = random.randint(1, context_window_size)
        # get a random allele before the center allele
        for target in index_alleles[max(0, index - context): index]:
            yield center, target
        # get a random allele after the center allele
        for target in index_alleles[index + 1: index + context + 1]:
            yield center, target

# Step 3: generate a training batch
def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window):
    """ Generate training batch pair samples. """
    alleles = read_data(FILE_NAME)
    single_gen = generate_sample(alleles, skip_window)
    return get_batch(single_gen, batch_size)

# if __name__ == '__main__':
#     t_alleles = read_data(FILE_NAME)
#     print(t_alleles)
#     print(type(t_alleles))
#     t_single_gen = generate_sample(t_alleles, 1)
#     t_batch = get_batch(t_single_gen, 128)
#     for i in t_batch:
#         print(i)
