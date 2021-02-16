# @author semese

import numpy as np
import torch
from torch import nn

from src.utils.nn_utils import ObjectDict, accuracy, print_train_info, save_model


class MLP(nn.Module):
    def __init__(self, hparams):
        """
        Multilayer feedforward models.

        :param dict hparams: Dictionary of models and hyper-parameters.
        """
        super(MLP, self).__init__()

        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        )

        # define layers
        self.hidden = nn.ModuleList()
        for i in range(len(self.hparams.dim_list) - 1):
            self.hidden.append(nn.Linear(self.hparams.dim_list[i], self.hparams.dim_list[i + 1]))

        # dropout prevents over-fitting of data
        if self.hparams.p_dropout:
            self.dropout = nn.Dropout(self.hparams.p_dropout)

        # define activation function
        self.relu = torch.nn.ReLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, feat_extr=False):
        """
        Forward pass.

        :param tensor x: Input tensor of size (batch_size, n_samples, n_features)
        :param bool feat_extr: If True, the output of the last hidden layer is returned instead of the network output.
        :return: Model output or values of last hidden layer.
        """
        # forward pass through the network
        for layer in self.hidden[:-1]:
            x = self.relu(layer(x))
            if self.hparams.p_dropout:
                x = self.dropout(x)

        if feat_extr:
            # extract the output of the last hidden layer
            return x
        else:
            # the last layer is a linear layer without activation function
            output = self.hidden[-1](x)

            return output


class MLPExtended(MLP):
    def __init__(self, hparams_dict):
        """
        Initializer for the extended multilayer feedforward models.

        :param hparams_dict: Dictionary of models and hyper-parameters.
        """
        super(MLPExtended, self).__init__(hparams_dict)

        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=False,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.loss_during_training = []

        self.valid_loss_during_training = []

        self.to(self.device)

    def train_loop(
            self,
            train_data,
            valid_data=None,
            verbose=True,
            print_every=10,
            model_path=None,
            save_all=False,
    ):
        """
        Method to train the models.

        :param tuple train_data: the training data set
        :param tuple valid_data: the validation set
        :param bool verbose: boolean indicator whether to print training info
        :param int print_every: how frequently to print the training details
        :param str model_path: full path where to save the models during training
        :param bool save_all: flag to indicate whether to save the models from each step or only when the valid_loss decreases
        :return:
        """
        # track change in validation loss
        if self.valid_loss_during_training:
            valid_loss_min = np.amin(self.valid_loss_during_training)
        else:
            valid_loss_min = np.Inf

        # SGD Loop
        for epoch in range(1, self.hparams.max_nb_epochs + 1):
            ###################
            # train the models #
            ###################
            self.train()

            # clear the gradients of all optimized variables
            self.optimiser.zero_grad()

            loss, output, target = self.forward_pass(train_data)

            # backward pass: compute gradient of the loss with respect to models parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            self.optimiser.step()

            # save training loss
            self.loss_during_training.append(loss.item())

            # training accuracy
            train_acc = accuracy(output, target, logits=True)

            ######################
            # validate the models #
            ######################
            self.eval()

            # validation loss
            v_loss, output, target = self.forward_pass(valid_data)

            # save validation loss
            self.valid_loss_during_training.append(v_loss.item())

            # validation accuracy
            valid_acc = accuracy(output, target, logits=True)

            # print training/validation statistics
            if verbose and epoch % print_every == 0:
                print_train_info(epoch, self.hparams.max_nb_epochs, loss, train_acc, v_loss, valid_acc)

            # save models if validation loss has decreased or save_all parameter is set
            valid_loss_min = save_model(self, epoch, v_loss, valid_loss_min, verbose, model_path, save_all)

    def forward_pass(self, data_set):
        """
        Perform forward pass on given dataset.

        :param tuple data_set: the training or validation set.
        :return: The evaluated loss function, network output and target labels.
        """
        # move tensors to GPU if CUDA is available
        data, target = (
            data_set[0].to(self.device),
            data_set[1].to(self.device),
        )

        # compute predicted outputs by passing inputs to the models
        output = self.forward(data)

        # return loss
        return self.criterion(output, target.long()), output, target

    def feature_extraction(self, data):
        """
        Extract the output of the last hidden layer of the network.

        :param tensor data: Data to perform forward pass on.
        :return: Output of the last hidden layer of the network.
        """
        self.eval()

        # move tensors to GPU if CUDA is available
        data = data.to(self.device)

        # compute predicted outputs by passing inputs to the models
        output = self.forward(data, feat_extr=True)

        return output.cpu().detach().numpy()
