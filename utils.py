# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import numpy
import torch.utils.data
import matplotlib.pyplot as plt

from os import path
from matplotlib.pyplot import Subplot


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


class LabelledDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset and its associated labels.

    @param dataset Numpy array representing the dataset.
    @param labels One-dimensional array of the same length as dataset with
           non-negative int values.
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]


def plot_with_labels(mat,
                     labels,
                     use_axes=None,
                     marker='o',
                     legend='',
                     title='',
                     save_path=None):
    # TODO: handle 3D plots
    # TODO: handle more than 7 labels

    # Prep plot
    all_lab = list( set(labels) )
    all_col = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    argout = [None, None]
    update = False
    leg_names = []
    if (use_axes is None) or (not type(use_axes) is Subplot):
        hF = plt.figure(figsize=[16,8])
        use_axes = hF.add_subplot(1, 1, 1)
        argout = [hF, use_axes]
    else:
        update = True
        leg_names = [x.get_text() for x in use_axes.legend_.get_texts()]

    for i_lab, i_col in zip(all_lab, all_col):
        i_pts = [x==i_lab for x in labels]
        use_axes.plot(mat[i_pts, 0], mat[i_pts, 1], marker=marker, color=i_col, linestyle='None', label=i_lab)

    leg_names += [legend+str(x) for x in all_lab]
    use_axes.legend(leg_names)
    if not update:
        use_axes.set_xlabel('dimension1')
        use_axes.set_ylabel('dimension2')
        use_axes.set_title(title)

    if (not save_path is None) and (title!=''):
        save_name = title.replace(' ', '_')+'.png'
        plt.savefig( path.join(save_path, save_name) )

    return argout
