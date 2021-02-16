# @author semese

import torch


# ------------------------------------------------- Model Training Utils --------------------------------------------- #
class ObjectDict(dict):
    """
    Interface similar to an argparser.
    """

    def __init__(self):
        super().__init__()

    def __setattr__(self, attr, value):
        self[attr] = value
        return self[attr]

    def __getattr__(self, attr):
        if attr.startswith("_"):
            # https://stackoverflow.com/questions/10364332/how-to-pickle-python-object-derived-from-dict
            raise AttributeError
        return dict(self)[attr]

    @property
    def __dict__(self):
        return dict(self)


def accuracy(y_probs, y, logits=False):
    """
    Accuracy classification score.

    :param tensor y_probs: Predicted labels, as returned by a classifier.
    :param tensor y: Ground truth (correct) labels.
    :param bool logits: If False, y_probs is the log-softmax output.
    :return: the computed accuracy
    """

    # If logits are given as output, pass them through a log-softmax
    if logits:
        y_probs = torch.log_softmax(y_probs, dim=1)

    # Get predictions from the maximum value
    _, y_hat = torch.max(y_probs.data, 1)

    # Total number of labels
    total = y.size(0)

    # Total correct predictions
    correct = (y_hat == y).sum()

    # Compute and return accuracy
    return (100.0 * correct / total).item()


def print_train_info(epoch, max_nb_epochs, loss, train_acc, v_loss, valid_acc):
    """
    Print loss and accuracy evaluation during training.

    :param int epoch: current epoch in training
    :param int max_nb_epochs: maximum number of epochs
    :param float loss: training loss
    :param float train_acc: training accuracy
    :param float v_loss: validation loss
    :param float valid_acc: validation accuracy
    :return:
    """
    print(
        "Epoch: {}/{}\t".format(epoch, max_nb_epochs),
        "Trn. Loss: {:.4f}\t".format(loss),
        "Trn. Acc: {:.2f}% \t".format(train_acc),
        "Val. Loss: {:.4f}\t".format(v_loss),
        "Val. Acc: {:.2f}%".format(valid_acc),
    )


def save_model(model, epoch, v_loss, valid_loss_min, verbose, model_path, save_all):
    """
    Save models if validation loss has decreased or save_all parameter is set.

    :param model: Pytorch network models.
    :param int epoch: current epoch in training.
    :param float v_loss: validation loss
    :param float valid_loss_min: minimum validation loss
    :param bool verbose: boolean indicator whether to print training info
    :param str model_path: full path where to save the models during training
    :param bool save_all: flag to indicate whether to save the models from each step or only when the valid_loss decreases
    :return: updated minimum validation loss
    """
    if (v_loss <= valid_loss_min) and (model_path is not None):
        if verbose:
            print(
                "Epoch: {}\t".format(epoch),
                "Validation loss decreased ({:.4f} --> {:.4f}).  Saving models ...".format(
                    valid_loss_min, v_loss
                ),
            )
        torch.save(model, model_path + "earl_stp.pt")
        valid_loss_min = v_loss

    if save_all and (model_path is not None):
        torch.save(model, model_path + "epoch_" + str(epoch) + ".pt")

    return valid_loss_min


# --------------------------------------------- RNN Training Utils --------------------------------------------------- #
def filter_out_padding_and_missing(logits, y, pad_token):
    """
    Before the negative log likelihood is computed, the activations have to be masked out,
    since padded & missing items in the output vector should not be considered.

    :param tensor logits: Predicted log-probabilities, as returned by a classifier.
    :param tensor y: Ground truth (correct) labels.
    :param int pad_token: Padding token used in the sequences.
    :return: Filtered log-probabilities and ground truth sequences.
    """

    # Create a mask by filtering out all tokens that ARE NOT the padding token or NaN
    not_nan_mask = ~torch.isnan(y) & (y != pad_token)

    # Return without missing or padded values
    return logits[not_nan_mask], y[not_nan_mask]