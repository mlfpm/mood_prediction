# @author semese

import numpy as np
import torch
from torch import nn

from src.utils.nn_utils import filter_out_padding_and_missing, ObjectDict, accuracy, print_train_info, save_model


class RNN(nn.Module):
    def __init__(self, hparams):
        """
        Recurrent neural network models.
        :param dict hparams: Dictionary of models and hyper-parameters.
        """
        super(RNN, self).__init__()

        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        )

        # Building the RNN
        # batch_first=True causes input/output tensors to have to be of shape
        # (batch_dim, seq_dim, feature_dim)
        if self.hparams.cell_type == "GRU":
            self.rnn = nn.GRU(
                self.hparams.input_dim,
                self.hparams.hidden_dim,
                self.hparams.n_layers,
                batch_first=True,
                dropout=self.hparams.drop_prob,
                bidirectional=self.hparams.bidirectional,
            )
        elif self.hparams.cell_type == "RNN":
            self.rnn = nn.RNN(
                self.hparams.input_dim,
                self.hparams.hidden_dim,
                self.hparams.n_layers,
                batch_first=True,
                nonlinearity="relu",
            )
        elif self.hparams.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.hparams.input_dim,
                hidden_size=self.hparams.hidden_dim,
                batch_first=True,  # The input and output tensors are provided as (batch, seq, feature)
                num_layers=self.hparams.n_layers,
                bidirectional=self.hparams.bidirectional,  # If True, becomes a bidirectional LSTM. Default: False
                dropout=self.hparams.drop_prob,
                # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                # with dropout probability equal to dropout. Default: 0
            )
            self.hidden_out_dim = self.hparams.hidden_dim * (
                    self.hparams.bidirectional + 1
            )
        else:
            raise NotImplementedError

        # Readout layer
        if self.hparams.cell_type == "LSTM":
            self.fc = nn.Linear(self.hidden_out_dim, int(self.hparams.output_dim))
        else:
            self.fc = nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, x_lengths):
        """
        Forward pass.
        :param tensor x: Input tensor of size (batch_size, seq_len, n_features)
        :param tensor x_lengths: Tensor containing the individual sequence lengths.
        :return: Model output.
        """
        batch_size, seq_len, _ = x.size()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_lengths.reshape(-1), batch_first=True, enforce_sorted=False
        )

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        if self.hparams.cell_type == "LSTM":
            (h0, c0) = self.init_hidden(batch_size)
            x, (_, _) = self.rnn(x, (h0.detach(), c0.detach()))
        else:
            h0 = self.init_hidden(batch_size)
            x, _ = self.rnn(x, h0.detach())

        # undo the packing operation
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=self.hparams.pad_token,
            total_length=seq_len,
        )

        # out.size() --> batch_size * seq_len, self.hparams.hidden_dim
        x = x.view(-1, x.size(2))

        # out.size() --> batch_size * seq_len, output_dim
        out = self.log_softmax(self.fc(x))

        return out

    def init_hidden(self, batch_size):
        """
        Initialize the hidden units of the RNN.
        :param int batch_size: Batch size.
        :return: Zero-initialized hidden units.
        """
        if self.hparams.cell_type == "LSTM":
            # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
            n_dir = 1 + int(self.hparams.bidirectional)
            h0 = torch.zeros(
                self.hparams.n_layers * n_dir,
                batch_size,
                self.hparams.hidden_dim,
                dtype=torch.float,
            ).requires_grad_().to(self.device)

            c0 = torch.zeros(
                self.hparams.n_layers * n_dir,
                batch_size,
                self.hparams.hidden_dim,
                dtype=torch.float,
            ).requires_grad_().to(self.device)

            return h0, c0
        else:
            h0 = torch.zeros(
                self.hparams.n_layers,
                batch_size,
                self.hparams.hidden_dim,
                dtype=torch.float,
            ).requires_grad_().to(self.device)
            return h0


class RNNExtended(RNN):
    def __init__(self, hparams_dict):
        """
        Initializer for the extended multilayer feedforward models.
        :param hparams_dict: Dictionary of models and hyper-parameters.
        """
        super(RNNExtended, self).__init__(hparams_dict)

        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=False,
        )

        self.criterion = nn.NLLLoss()

        self.loss_during_training = []

        self.valid_loss_during_training = []

        self.to(self.device)

    def train_loop(
            self,
            train_loader,
            valid_loader=None,
            verbose=True,
            print_every=10,
            model_path=None,
            save_all=False,
    ):
        """
        Method to train the models.
        :param DataLoader train_loader: the training data set
        :param DataLoader valid_loader: the validation set
        :param bool verbose: boolean indicator whether to print training info
        :param int print_every: how frequently to print the training details
        :param str model_path: ull path where to save the models during training
        :param bool save_all: flag to indicate whether to save the models from
            each step or only when the valid_loss decreases
        :return:
        """
        # track change in validation loss
        if self.valid_loss_during_training:
            valid_loss_min = np.amin(self.valid_loss_during_training)
        else:
            valid_loss_min = np.Inf

        # SGD Loop
        for epoch in range(1, self.hparams.max_nb_epochs + 1):
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            ###################
            # train the models #
            ###################
            self.train()
            # Batch loop
            for data, target, lengths in train_loader:
                # clear the gradients of all optimized variables
                self.optimiser.zero_grad()

                # forward pass
                loss, outputs_i, y_i = self.forward_pass((data, target, lengths))

                # backward pass: compute gradient of the loss with respect to models parameters
                loss.backward()

                if not (self.hparams.clip is None):
                    nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip)

                # perform a single optimization step (parameter update)
                self.optimiser.step()

                # update training loss
                train_loss += loss.item() * data.size(0)

                # calculate the accuracy
                train_acc += accuracy(outputs_i, y_i) * data.size(0)

            ######################
            # validate the models #
            ######################
            self.eval()
            for data, target, lengths in valid_loader:
                # forward pass
                loss, outputs_i, y_i = self.forward_pass((data, target, lengths))

                # update average validation loss
                valid_loss += loss.item() * data.size(0)

                # calculate the accuracy
                valid_acc += accuracy(outputs_i, y_i) * data.size(0)

            # calculate average losses
            train_loss = train_loss / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)

            # save losses
            self.loss_during_training.append(train_loss)
            self.valid_loss_during_training.append(valid_loss)

            # calculate average accuracies
            train_acc = train_acc / len(train_loader.sampler)
            valid_acc = valid_acc / len(valid_loader.sampler)

            # print training/validation statistics
            if verbose and epoch % print_every == 0:
                print_train_info(epoch, self.hparams.max_nb_epochs, train_loss, train_acc, valid_loss, valid_acc)

            # save models if validation loss has decreased or save_all parameter is set
            valid_loss_min = save_model(self, epoch, valid_loss, valid_loss_min, verbose, model_path, save_all)

    def forward_pass(self, data_set):
        """
        Perform forward pass on given dataset.
        :param tuple data_set: the training or validation set.
        :return: The evaluated loss function, network output and target labels.
        """
        # move tensors to GPU if CUDA is available
        data, target, lengths = (
            data_set[0].to(self.device),
            data_set[1].to(self.device),
            data_set[2].to(self.device),
        )

        # forward pass: compute predicted outputs by passing inputs to the models
        output = self.forward(data, lengths)

        # Mask for the missing labels and padding
        outputs_i, y_i = filter_out_padding_and_missing(
            output, target.view(-1), self.hparams.pad_token
        )

        # calculate the batch loss
        loss = self.criterion(outputs_i, y_i.long())

        return loss, outputs_i, y_i

    def predict_for_eval(self, test_loader):
        """
        Function to compute predictions and return the results without the padding
        and missing parts, so that the evaluation metrics can then be computed on it directly.
        :param DataLoader test_loader: Data loader for the evaluation set.
        :return: List of predicted labels, target labels and prediction probabilities.
        """
        y_true_list = []
        y_pred_list = []
        y_score_list = []

        self.eval()
        for i, (data, target, lengths) in enumerate(test_loader):
            # move tensors to GPU if CUDA is available
            data, target, lengths = (
                data.to(self.device),
                target.to(self.device),
                lengths.to(self.device),
            )

            # forward pass: compute predicted outputs by passing inputs to the models
            output = self.forward(data, lengths)

            # mask for the missing labels and padding
            y_probs, y = filter_out_padding_and_missing(
                output, target.view(-1), self.hparams.pad_token
            )

            # Get predictions from the maximum value
            _, y_hat = torch.max(y_probs.data, 1)

            y_true_list.append(y.cpu().numpy().reshape(-1))
            y_pred_list.append(y_hat.cpu().numpy().reshape(-1))
            y_score_list.append(y_probs.cpu().detach().numpy())

        return (
            np.hstack(y_true_list),
            np.hstack(y_pred_list),
            np.array(
                [
                    y_score_list[i][j]
                    for i in range(len(y_score_list))
                    for j in range(len(y_score_list[i]))
                ]
            ),
        )
