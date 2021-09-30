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


import math
import os
import json
import numpy as np
import torch
import sklearn
import sklearn.svm
from sklearn import metrics
import sklearn.model_selection

from src.unsupervised import utils, \
                             losses, \
                            networks

from torch.utils.tensorboard import SummaryWriter



class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    """
    "Virtual" class to wrap an encoder of time series as a PyTorch module and
    a SVM classifier with RBF kernel on top of its computed representations in
    a scikit-learn class.

    All inheriting classes should implement the get_params and set_params
    methods, as in the recommendations of scikit-learn.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param encoder Encoder PyTorch module.
    @param params Dictionaries of the parameters of the encoder.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 network, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.i_step = 0
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.network = network
        self.encoder = network.encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_channel = losses.channel_triplet_loss.ChannelTripletLoss(compared_length, nb_random_samples, negative_penalty)
        self.loss_channel_varying = losses.channel_triplet_loss.ChannelTripletLossVaryingLength(compared_length, nb_random_samples, negative_penalty)
        self.loss_combined = losses.combined_triplet_loss.CombinedTripletLoss(compared_length, nb_random_samples, negative_penalty)
        self.loss_combined_varying = losses.combined_triplet_loss.CombinedTripletLossVaryingLength(compared_length, nb_random_samples, negative_penalty)
        self.loss = losses.triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        # Could be beneficial to have different learning rate for each optimizer
        self.optimizer_supervised = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.writer=SummaryWriter() #tensorboard logging
        # Add model saving folder
        os.makedirs( os.path.join(self.writer.log_dir, 'saved_models'), exist_ok=True )

    def save_encoder(self, prefix_file):
        """
        Saves the encoder.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture.replace('withClassifier', '')  + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the network containing both the encoder and the classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if not os.path.exists(os.path.split(prefix_file)[0]):
            os.makedirs(os.path.split(prefix_file)[0])
        self.save_encoder(prefix_file)
        torch.save(
            self.network.state_dict(),
            prefix_file + '_' + self.architecture + '.pth'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture.replace('withClassifier', '') + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture.replace('withClassifier', '') + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))
    
    def load_encoder_from_path(self, path):
        """
        Loads an encoder.

        @param Path of the file where the model should
               be loaded (e.g. 'path/to/vitals_CausalCNN_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                path,
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                path,
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads the network containing an encoder and a classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture).pth').
        """
        if self.cuda:
            self.network.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.network.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '.pth',
                map_location=lambda storage, loc: storage
            ))

    def fit_classifier(self, X, y, X_valid=None, y_valid=None, save_memory=False, verbose=False, freeze_encoder=False):
        """
        Supervised Training.
        Can freeze encoder and train only classifier or can train both the encoder and the classifer.

        """
        print(f'Training classifier with freeze_encoder= {freeze_encoder}')

        if freeze_encoder:
            # Freeze Encoder
            for param in self.encoder.children():
                param.requires_grad = False
        
        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        # # Weights classes to try to fix only predicting the majority class
        label_distrib = np.unique(y, return_counts=True)
        class_sizes = label_distrib[1][~np.isnan(label_distrib)[0]]
        smallest_class_size = np.min(class_sizes)
        weights = smallest_class_size / torch.tensor(class_sizes, dtype=torch.double)
        if self.cuda:
            weights = weights.cuda(self.gpu)

        criterion_supervised =  torch.nn.CrossEntropyLoss(weight=weights)
        # criterion_supervised =  torch.nn.CrossEntropyLoss()

        # remove examples with nan label
        nan_mask = np.isnan(y)
        if nan_mask.any(): # if there is any nan label
            if nan_mask.all(): # if there are all nan, go to the next batch
                sys.exit('CANNOT DO SUPERVISED TRAINING WITH ALL LABELS AS NAN')
            indices = np.nonzero(~nan_mask)[0]
            supervised_X = X[indices]
            supervised_y = y[indices]
            train_torch_dataset = utils.LabelledDataset(supervised_X,supervised_y)

        else:
            train_torch_dataset = utils.LabelledDataset(X,y)

        nb_classes = np.shape(np.unique(y, return_counts=True)[1])[0]
        train_size = np.shape(X)[0]
        ratio = train_size // nb_classes
        if varying:
            print("BATCH SIZE CHANGED TO 1")

        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size if not varying else 1, shuffle=True
        )

        max_score = 0
        # max_auc = 0
        # max_f1 = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        running_loss = 0.0
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        best_network = None
        found_best = False

        # training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch, batch_label in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                    batch_label = batch_label.cuda(self.gpu)
                
                self.optimizer_supervised.zero_grad()
                if varying:
                    with torch.no_grad():
                        # remove nan padding at the end
                        length = batch.size(2) - torch.sum(torch.isnan(batch[0, 0]))
                        batch = batch[:, :, :length]

                outputs = self.network(batch)
                batch_label = batch_label.long()
                loss = criterion_supervised(outputs, batch_label)
                loss.backward()
                self.optimizer_supervised.step()
                i += 1

                running_loss += loss.item()
                frequency = round(len(X)/10)
                if frequency < 10:
                    frequency = 10
                if (i % frequency) == (frequency-1):  # every n mini-batches
                    if freeze_encoder:
                        prefix='Classifier'
                    else:
                        prefix='Network'
                    
                    if nan_mask.any(): #if contains missing labels, use only those with labels to evaluate the accuracy and other measures
                        best_network, found_best, count, max_score = self.log(running_loss, frequency, i, supervised_X, supervised_y, varying, X_valid, y_valid, ratio, train_size, count, max_score, found_best, best_network, prefix=prefix)
                    else:
                        best_network, found_best, count, max_score = self.log(running_loss, frequency, i, X, y, varying, X_valid, y_valid, ratio, train_size, count, max_score, found_best, best_network, prefix=prefix)
                    running_loss = 0.0
                if i >= self.nb_steps or count == self.early_stopping:
                    break
            epochs += 1

            if count >= self.early_stopping:
                break

        if freeze_encoder:
            # Unfreeze Encoder
            for param in self.encoder.children():
                param.requires_grad = True
        
        # If a better model was found, use it
        if found_best:
            self.network = best_network

        return self.network


    def fit_encoder(self, X, y=None, save_memory=False, verbose=False, training_type='length'):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """

        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        if varying:
            if training_type == 'channel':
                loss_fn = self.loss_channel_varying
            elif training_type == 'combined':
                loss_fn = self.loss_combined_varying
            else: # training_type == 'length'
                loss_fn = self.loss_varying
        else: # not varying
            if training_type == 'channel':
                loss_fn = self.loss_channel
            elif training_type == 'combined':
                loss_fn = self.loss_combined
            else: # training_type == 'length'
                loss_fn = self.loss_varying        

        train = torch.from_numpy(X) # The returned tensor and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. 
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        # only used for early spotting i.e. svm supervised training. Therefore, remove examples with nan label
        if y is not None:
            nan_mask = np.isnan(y)
            if nan_mask.any(): # if there is any nan label
                if nan_mask.all(): # if there are all nan, go to the next batch
                    sys.exit('CANNOT VALIDATE AND DO EARLY STOPPING WITHOUT LABELS, HERE ALL LABELS ARE NAN')
                indices = np.nonzero(~nan_mask)[0]
                supervised_X = X[indices]
                supervised_y = y[indices]
            else:
                supervised_X = X
                supervised_y = y

        max_score = 0
        max_auc = 0
        max_f1 = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        running_loss = 0.0
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False
        loss_fn.count = 0
        best_loss = np.inf

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = loss_fn(batch, self.encoder, train, save_memory=save_memory)
                loss.backward()
                self.optimizer.step()
                i += 1
                # log the running loss
                self.writer.add_scalar('encoder training loss', loss.item(), i)
                running_loss += loss.item()
                # Update best encoder
                if loss.item()<best_loss:
                    best_loss = loss.item()
                    save_path = os.path.join(self.writer.log_dir, 'saved_models', 'best')
                    self.save_encoder(save_path)

                frequency = round(len(train)/10)
                if frequency < 10:
                    frequency = 10
                if (i % frequency) == (frequency-1):  # every 1000 mini-batches
                    name = 'Svm Training'

                    # Save model
                    save_path = os.path.join(self.writer.log_dir, 'saved_models', 'step_'+str(i))
                    self.save_encoder(save_path)

                    if y is not None:
                        features = self.encode(supervised_X)
                        features_train, features_test = np.array_split(features, 2)
                        supervised_y_train, supervised_y_test = np.array_split(supervised_y, 2)
                        classifier = self.fit_svm(features_train, supervised_y_train)
                        print('previous max accuracy:', max_score)

                        y_pred = classifier.predict(features_test)
                        accuracy = metrics.accuracy_score(supervised_y_test, y_pred)
                        if self.nb_class == 2:
                            auc = metrics.roc_auc_score(supervised_y_test, y_pred)
                            f1 = metrics.f1_score(supervised_y_test, y_pred, zero_division=1)
                            precision = metrics.precision_score(supervised_y_test, y_pred)
                            recall = metrics.recall_score(supervised_y_test, y_pred)
                        else:
                            auc = -1
                            f1 = metrics.f1_score(supervised_y_test, y_pred, zero_division=1, average='weighted')
                            precision = metrics.precision_score(supervised_y_test, y_pred, average='weighted')
                            recall = metrics.recall_score(supervised_y_test, y_pred, average='weighted')
                        print('label predicted distribution')
                        print(np.unique(y_pred, return_counts=True))
                        print('real label distribution')
                        print(np.unique(supervised_y_test, return_counts=True))
                        print(f'{name} stats     accuracy: {accuracy}, auc: {auc}, f1: {f1}, precision: {precision}, recall: {recall}')

                        self.writer.add_scalar(name+' accuracy', accuracy, i * frequency)

                        # self.writer.add_scalar(name+' auc', auc, i * frequency)
                        self.writer.add_scalar(name+' f1', f1, i * frequency)
                        running_loss = 0.0

                    # Early stopping strategy
                    if (self.early_stopping is not None) and (y is not None): #and (ratio >= 5 and train_size >= 50):

                        count += 1
                        # if auc_score > max_auc:
                        #     max_auc = auc_score
                        if f1 > max_f1:
                            max_f1 = f1
                        # if accuracy > max_score:
                        #     max_score = accuracy
                            count = 0
                            found_best = True
                            best_encoder = type(self.encoder)(**self.params)
                            best_encoder.double()
                            if self.cuda:
                                best_encoder.cuda(self.gpu)
                            best_encoder.load_state_dict(self.encoder.state_dict())

                    # Baseline
                    if y is not None:
                        from sklearn.dummy import DummyClassifier
                        dummy_clf = DummyClassifier(strategy="stratified") # 'stratified' most_frequent"
                        dummy_clf = dummy_clf.fit(features_train, supervised_y_train)
                        y_pred = dummy_clf.predict(features_test)

                        accuracy = metrics.accuracy_score(supervised_y_test, y_pred)
                        if self.nb_class == 2:
                            auc = metrics.roc_auc_score(supervised_y_test, y_pred)
                            f1 = metrics.f1_score(supervised_y_test, y_pred, zero_division=1)
                            precision = metrics.precision_score(supervised_y_test, y_pred)
                            recall = metrics.recall_score(supervised_y_test, y_pred)
                        else:
                            auc = -1
                            f1 = metrics.f1_score(supervised_y_test, y_pred, zero_division=1, average='weighted')
                            precision = metrics.precision_score(supervised_y_test, y_pred, average='weighted')
                            recall = metrics.recall_score(supervised_y_test, y_pred, average='weighted')
                        print(f'Dummy stats     accuracy: {accuracy}, auc: {auc}, f1: {f1}, precision: {precision}, recall: {recall}')

                if i >= self.nb_steps or (self.early_stopping is not None and count >= self.early_stopping):
                    break
            epochs += 1
            if (self.early_stopping is not None) and (count >= self.early_stopping):
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.encoder

    def fit_svm(self, features, y):
        """
        ONLY USE FOR EARLY STOPPING OF UNSUPERVISED TRAINING
        Trains an SVM classifier (with RBF kernel) using precomputed features.
        @param features Computed features of the training set.
        @param y Training labels.
        """

        nb_classes = np.shape(np.unique(y, return_counts=True)[1])[0]
        train_size = np.shape(features)[0]

        classifier = sklearn.svm.SVC(
            C=1 / self.penalty
            if self.penalty is not None and self.penalty > 0
            else np.inf,
            gamma='scale',
            class_weight='balanced'
        )
        if train_size // nb_classes < 5 or train_size < 50:
            return classifier.fit(features, y)
        else:
            if self.penalty is None:
                grid_search = sklearn.model_selection.GridSearchCV(
                    classifier, {
                        'C': [
                            0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                            np.inf
                        ],
                        'kernel': ['rbf'],
                        'degree': [3],
                        'gamma': ['scale'],
                        'coef0': [0],
                        'shrinking': [True],
                        'probability': [False],
                        'tol': [0.001],
                        'cache_size': [200],
                        'class_weight': [None],
                        'verbose': [False],
                        'max_iter': [10000000],
                        'decision_function_shape': ['ovr'],
                        'random_state': [None],
                        'class_weight': ['balanced']
                    },
                    cv=5,  n_jobs=5 
                )
                if train_size <= 10000:
                    grid_search.fit(features, y)
                else:
                    # If the training set is too large, subsample 10000 train
                    # examples
                    split = sklearn.model_selection.train_test_split(
                        features, y,
                        train_size=10000, random_state=0, stratify=y
                    )
                    grid_search.fit(split[0], split[2])
                classifier = grid_search.best_estimator_
        return classifier


    def fit(self, X, y, X_valid=None, y_valid=None, save_memory=False, verbose=False, missing=0.0):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        print("Parameters")
        print(json.dumps(self.get_params(), indent=4))

        ## Fitting encoder with unsupervised training on channels
        if self.seq_encoder_channel:
            self.encoder = self.fit_encoder(X, y=y, save_memory=save_memory, verbose=verbose, training_type='channel')
            print('Sequentially trained channels encoder')

        # if sequential train encoder: (i.e. unsupervised)
        ## Fitting encoder
        if self.seq_encoder:
            self.encoder = self.fit_encoder(X, y=y, save_memory=save_memory, verbose=verbose, training_type='length')
            print('Sequentially trained encoder')

        ## Fitting encoder with unsupervised training on both length and channel
        if self.seq_encoder_combined:
            self.encoder = self.fit_encoder(X, y=y, save_memory=save_memory, verbose=verbose, training_type='combined')
            print('Sequentially combined trained encoder')
        
        # random indices to consider as missing
        if missing != 0.0:
            indices_to_keep = np.random.choice(X.shape[0], int(X.shape[0]*(1-missing)), replace=False)
            indices_missing = np.array([e for e in range(X.shape[0]) if e not in indices_to_keep])
            X_without_missing = X[indices_to_keep,:,:]
            y_without_missing = y[indices_to_keep]
            y[indices_missing] = -99999


        # if sequential train classifier
        if self.seq_classifier:
            freeze_encoder = True
            if missing != 0.0:
                self.network = self.fit_classifier(X_without_missing, y_without_missing, X_valid, y_valid, save_memory, verbose, freeze_encoder=freeze_encoder)
            else:
                self.network = self.fit_classifier(X, y, X_valid, y_valid, save_memory, verbose, freeze_encoder=freeze_encoder)
            print('Sequentially trained classifier')
        
        # if sequential train whole network
        if self.seq_network:
            freeze_encoder = False
            if missing != 0.0:
                self.network = self.fit_classifier(X_without_missing, y_without_missing, X_valid, y_valid, save_memory, verbose, freeze_encoder=freeze_encoder)
            else:
                self.network = self.fit_classifier(X, y, X_valid, y_valid, save_memory, verbose, freeze_encoder=freeze_encoder)
            print('Sequentially trained the whole network')

        # if doesn't alternate training
        if self.alt_encoder == -1 and self.alt_classifier == -1 and self.alt_network == -1 and self.alt_encoder_channel == -1:
            return self

        # Training in alternance
        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        criterion_supervised =  torch.nn.CrossEntropyLoss()

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = np.shape(np.unique(y, return_counts=True)[1])[0]
            train_size = np.shape(X)[0]
            ratio = train_size // nb_classes

        #train_torch_dataset = utils.Dataset(X)
        train_torch_dataset = utils.LabelledDataset(X,y)

        if varying:
            print("BATCH SIZE CHANGED TO 1")
        # Batch size = 1 if varying length
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size if not varying else 1, shuffle=True
        )

        max_score = 0
        max_auc = 0
        max_f1 = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        best_network = None
        running_loss = 0.0
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch, batch_label in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                    batch_label = batch_label.cuda(self.gpu)

                self.optimizer.zero_grad()
                self.optimizer_supervised.zero_grad()


                if self.alt_encoder != -1 and (i % self.alt_encoder) == 0:
                    # Unsupervised step
                    if not varying:
                        loss = self.loss(
                            batch, self.encoder, train, save_memory=save_memory
                        )
                    else:
                        # already batch_size=1 to deal with varying length, but need to remove Nan padding at the end
                        loss = self.loss_varying(
                            batch, self.encoder, train, save_memory=save_memory
                        )
                    loss.backward()
                    self.optimizer.step()
                    # Unsupervised step

                if self.alt_encoder_channel != -1 and ((i % self.alt_encoder_channel) == 0):
                    # Unsupervised step
                    if not varying:
                        loss = self.loss_channel(
                            batch, self.encoder, train, save_memory=save_memory
                        )
                    else:
                        # already batch_size=1 to deal with varying length, but need to remove Nan padding at the end
                        loss = self.loss_channel_varying(
                            batch, self.encoder, train, save_memory=save_memory
                        )
                    loss.backward()
                    self.optimizer.step()
                    # Unsupervised step
                
                if ((batch_label == -99999).any()):
                    keep = (batch_label != -99999)
                    batch = batch[keep]
                    batch_label = batch_label[keep]
                    if batch.shape[0] == 0:
                        continue

                if self.alt_classifier != -1 and (i % self.alt_classifier) == 0 :
                    
                    # Freeze Encoder
                    for param in self.encoder.children():
                        param.requires_grad = False
                    
                    # supervised step
                    if varying:
                        with torch.no_grad():
                            # already batch_size=1 to deal with varying length, but still contains Nan padding at the end
                            length = batch.size(2) - torch.sum(torch.isnan(batch[0, 0]))
                            batch = batch[:, :, :length] # remove nan padding

                    
                    outputs = self.network(batch)
                    loss = criterion_supervised(outputs, batch_label)
                    loss.backward()
                    self.optimizer_supervised.step()
                    # supervised step

                    # Unfreeze Encoder
                    for param in self.encoder.children():
                        param.requires_grad = True
                
                if self.alt_network != -1 and (i % self.alt_network) == 0:

                    # supervised step
                    if varying:
                        with torch.no_grad():
                            # already batch_size=1 to deal with varying length, but still contains Nan padding at the end
                            length = batch.size(2) - torch.sum(torch.isnan(batch[0, 0]))
                            batch = batch[:, :, :length] # remove nan padding
                    
                    outputs = self.network(batch)
                    loss = criterion_supervised(outputs, batch_label)
                    loss.backward()
                    self.optimizer_supervised.step()
                    # supervised step

                i += 1
                running_loss += loss.item()
                frequency = round(len(X)/10)
                if frequency < 10:
                    frequency = 10
                if (i % frequency) == (frequency-1):  # every n mini-batches
                    #log training and validation
                    prefix = 'Semi-supervised'
                    best_network, found_best, count, max_score = self.log(running_loss, frequency, i, X, y, varying, X_valid, y_valid, ratio, train_size, count, max_score, found_best, best_network, prefix=prefix)
                    running_loss = 0.0

                if i >= self.nb_steps or count == self.early_stopping:
                    break
            epochs += 1

            if count >= self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.network = best_network

        return self

    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = np.zeros((np.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        """
        features = np.empty((
                np.shape(X)[0], self.out_channels,
                np.shape(X)[2] - window + 1
        ))
        masking = np.empty((
            min(window_batch_size, np.shape(X)[2] - window + 1),
            np.shape(X)[1], window
        ))
        for b in range(np.shape(X)[0]):
            for i in range(math.ceil(
                (np.shape(X)[2] - window + 1) / window_batch_size)
            ):
                for j in range(
                    i * window_batch_size,
                    min(
                        (i + 1) * window_batch_size,
                        np.shape(X)[2] - window + 1
                    )
                ):
                    j0 = j - i * window_batch_size
                    masking[j0, :, :] = X[b, :, j: j + window]
                features[
                    b, :, i * window_batch_size: (i + 1) * window_batch_size
                ] = np.swapaxes(
                    self.encode(masking[:j0 + 1], batch_size=batch_size), 0, 1
                )
        return features

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        #features = np.zeros((np.shape(X)[0], self.out_channels))
        features = torch.empty((np.shape(X)[0], self.nb_class))

        self.network = self.network.eval()

        count = 0

        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                        features = features.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.network(batch) #.cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                        features = features.cuda(self.gpu)
                    # remove nan padding
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ) #.data.cpu().numpy()
                    features[count: count + 1] = self.network(
                        batch[:, :, :length]
                    ) #.cpu()
                    count += 1
        _, y_pred = torch.max(features.data, 1)
        self.network = self.network.train()
        return y_pred.cpu()

    def score(self, X, y, batch_size=50):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        y_pred = self.predict(X, batch_size=batch_size)
        return metrics.accuracy_score(y, y_pred)

    def score_all(self, X, y, batch_size=50):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        y_pred = self.predict(X, batch_size=batch_size)
        accuracy = metrics.accuracy_score(y, y_pred)
        if self.nb_class == 2:
            auc = metrics.roc_auc_score(y, y_pred)
            f1 = metrics.f1_score(y, y_pred, zero_division=1)
            precision = metrics.precision_score(y, y_pred)
            recall = metrics.recall_score(y, y_pred)
        else:
            auc = -1 #metrics.roc_auc_score(y, y_pred, multi_class='ovo')
            f1 = metrics.f1_score(y, y_pred, zero_division=1, average='weighted')
            precision = metrics.precision_score(y, y_pred, average='weighted')
            recall = metrics.recall_score(y, y_pred, average='weighted')
        print('label predicted distribution')
        print(torch.unique(y_pred, return_counts=True))

        return accuracy, auc, f1, precision, recall

    def log(self, running_loss, frequency, i, X, y, varying, X_valid, y_valid, ratio, train_size, count, max_score, found_best, best_network, prefix=''):
        name = (prefix+ ' Training').strip(' ')
        self.writer.add_scalar(name + ' loss', running_loss / frequency, i * frequency)
        accuracy, auc, f1, precision, recall = self.score_all(X, y,
                                                                batch_size=self.batch_size if not varying else 1)
        print(f'{name} stats     accuracy: {accuracy}, auc: {auc}, f1: {f1}, precision: {precision}, recall: {recall}, loss: {running_loss / frequency}')

        self.writer.add_scalar(name+' accuracy', accuracy, i * frequency)
        self.writer.add_scalar(name+' auc', auc, i * frequency)
        self.writer.add_scalar(name+' f1', f1, i * frequency)
        self.writer.add_scalar(name+' precision', precision, i * frequency)
        self.writer.add_scalar(name+' recall', recall, i * frequency)

        if X_valid is not None and y_valid is not None:
            name = (prefix+ ' Validation').strip(' ')
            accuracy, auc, f1, precision, recall = self.score_all(X_valid, y_valid,
                                                                batch_size=self.batch_size if not varying else 1)
            print(f'{name} stats     accuracy: {accuracy}, auc: {auc}, f1: {f1}, precision: {precision}, recall: {recall}')

            self.writer.add_scalar(name+' accuracy', accuracy, i * frequency)
            self.writer.add_scalar(name+' auc', auc, i * frequency)
            self.writer.add_scalar(name+' f1', f1, i * frequency)
            self.writer.add_scalar(name+' precision', precision, i * frequency)
            self.writer.add_scalar(name+' recall', recall, i * frequency)

            # Early stopping strategy
            if self.early_stopping is not None:
                count += 1
                # If the model is better than the previous one, update
                if f1 > max_score:
                # if accuracy > max_score:
                    count = 0
                    found_best = True
                    max_score = f1
                    # max_score = accuracy
                    best_network = type(self.network)(**self.params)
                    best_network.double()
                    if self.cuda:
                        best_network.cuda(self.gpu)
                    best_network.load_state_dict(self.network.state_dict())
        print(f'count={count}')
        return best_network, found_best, count, max_score



class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0, nb_class=2):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu, nb_class),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, nb_class),
            in_channels, out_channels, cuda, gpu, 
        )
        self.architecture = 'CausalCNNwithClassifier'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size
        self.nb_class = nb_class


    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu, nb_class):
        network = networks.causal_cnn.CausalCNNEncoderWithClassifier(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size, nb_class
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, nb_class):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'nb_class': nb_class
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(np.isnan(np.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        length = np.shape(X)[2]
        features = np.full(
            (np.shape(X)[0], self.out_channels, length), np.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, nb_class): 
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu, nb_class
        )
        return self
