"""
Created on Nov 20, 2019

@author: semese

Parts of the code come from: https://github.com/hmmlearn/hmmlearn
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from scipy import linalg
from scipy.special import logsumexp

plt.style.use("seaborn-ticks")

COVARIANCE_TYPES = frozenset(("spherical", "tied", "diagonal", "full"))
COLORS = sns.color_palette("colorblind", n_colors=15)


# ---------------------------------------------- Utils for the HMM models -------------------------------------------- #
def normalise(a, axis=None):
    """
    Normalise the input array so that it sums to 1.

    :param np.ndarray a: input data to be normalised
    :param int axis: dimension along which normalisation has to be performed
    :return: normalised array
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        a_sum[a_sum == 0] = 1  # Make sure it's not divided by zero.
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape
    a /= a_sum
    return a


def log_normalise(a, axis=None):
    """
    Normalise the input array so that ``sum(exp(a)) == 1``.

    :param np.ndarray a: input data to be normalised
    :param int axis: dimension along which normalisation has to be performed
    :return: normalised array
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state HMM in the degenerate case normalising a single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = logsumexp(a, axis, keepdims=True)
        a -= a_lse


def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalised to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this un-harmful warning.

    :param np.ndarray a: input data to be normalised
    :return: log-array
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def check_if_attributes_set(model, attr=None):
    """
    Checks if the models attributes are set before training. This is only necessary
    if the 'no_init' option is selected, because in that case the models parameters
    are expected to be set apriori, and won't be reinitialised before training or
    if some of the discrete emission probabilities aren't trained.

    :param model: defined HMM models
    :param str attr: attributes of the models. Can contain any combination of 's' for starting probabilities (pi),
        't' for transition matrix, and other characters for subclass-specific emission parameters.
    """

    params_dict = {"t": "A", "s": "pi", "e": "B", "m": "means", "c": "covars"}
    model_dict = model.__dict__
    if attr is not None:
        if not params_dict[attr] in model_dict.keys():
            raise AttributeError(
                "Attr self."
                + params_dict[attr]
                + " must be initialised before training"
            )
    else:
        for par in model.params:
            if params_dict[par] in model_dict.keys():
                continue
            else:
                raise AttributeError(
                    "Attr self."
                    + params_dict[par]
                    + " must be initialised before training"
                )


# Copied from scikit-learn 0.19.
def validate_covars(covars, covariance_type, n_states):
    """Do basic checks on matrix covariance sizes and values."""

    if covariance_type == "spherical":
        if len(covars) != n_states:
            raise ValueError("'spherical' covars have length n_states")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif covariance_type == "tied":
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif not np.allclose(covars, covars.T) or np.any(linalg.eigvalsh(covars) <= 0):
            raise ValueError("'tied' covars must be symmetric, " "positive-definite")
    elif covariance_type == "diagonal":
        if len(covars.shape) != 2:
            raise ValueError("'diagonal' covars must have shape " "(n_states, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diagonal' covars must be non-negative")
    elif covariance_type == "full":
        if len(covars.shape) != 3:
            raise ValueError(
                "'full' covars must have shape " "(n_states, n_dim, n_dim)"
            )
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError(
                "'full' covars must have shape " "(n_states, n_dim, n_dim)"
            )
        for n, cv in enumerate(covars):
            if not np.allclose(cv, cv.T) or np.any(linalg.eigvalsh(cv) <= 0):
                raise ValueError(
                    "component %d of 'full' covars must be "
                    "symmetric, positive-definite" % n
                )
    else:
        raise ValueError(
            "covariance_type must be one of "
            + "'spherical', 'tied', 'diagonal', 'full'"
        )
    return


def init_covars(tied_cv, covariance_type, n_states):
    """
        Helper function for initialising the covariances based on the
        covariance type. See class definition for details.
    """
    cv = None
    if covariance_type == "spherical":
        cv = tied_cv.mean() * np.ones((n_states,))
    elif covariance_type == "tied":
        cv = tied_cv
    elif covariance_type == "diagonal":
        cv = np.tile(np.diag(tied_cv), (n_states, 1))
    elif covariance_type == "full":
        cv = np.tile(tied_cv, (n_states, 1, 1))
    return cv


def fill_covars(covars, covariance_type="full", n_states=1, n_features=1):
    """
    Return the covariance matrices in full form: (n_states, n_features, n_features)
    """
    new_covars = np.array(covars, copy=True)
    if covariance_type == "full":
        return new_covars
    elif covariance_type == "diagonal":
        return np.array(list(map(np.diag, new_covars)))
    elif covariance_type == "tied":
        return np.tile(new_covars, (n_states, 1, 1))
    elif covariance_type == "spherical":
        eye = np.eye(n_features)[np.newaxis, :, :]
        new_covars = new_covars[:, np.newaxis, np.newaxis]
        temp = eye * new_covars
        return temp


# --------------------------------------------- Visualisation utils -------------------------------------------------- #
def plot_log_likelihood_evolution(log_likelihoods, transform_list=False):
    def log_like_list_to_df(logl_list):
        """
        Transform log-likelihood lists into a dataframe.

        :param list logl_list: list of list of log-likelihoods over training
        :return: a dataframe formed of the log-likelihoods over the initialisations and iterations
        """
        rows = []
        for init, logl_in in enumerate(logl_list):
            for it in range(len(logl_in)):
                rows.append(["init_" + str(init + 1), it, logl_in[it]])
        return pd.DataFrame(
            rows, columns=["initialisation", "iteration", "log_likelihood"]
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    if transform_list:
        sns.lineplot(
            x="iteration",
            y="log_likelihood",
            hue="initialisation",
            data=log_like_list_to_df(log_likelihoods),
            ax=ax,
        )
    else:
        ax.plot(np.arange(1, len(log_likelihoods[-1]) + 1), log_likelihoods[-1])
        ax.set_xlabel("# iterations")
        ax.set_ylabel("Log-likelihood")


# ------------------------------------------------------ IO Utils ---------------------------------------------------- #
def save_model(model, filename):
    """
    Save an HMM models to a pickle file.

    :param model: the models to be saved
    :param str filename: full path or just file name where to save the models
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    """
    Load an HMM models from a pickle file.

    :param str filename: full path or just file name where the models is saved
    :return: the models from the pickle file
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def pretty_print_hmm(model, hmm_type="Multinomial", states=None, emissions=None):
    """
    Pretty print the parameters of an HMM models.

    :param model: and HMM object
    :param str hmm_type: the type of the HMM models; can be "Multinomial", "Gaussian" or "Heterogeneous"
    :param list states: list with the name of states, if any
    :param list emissions: list of the names of the emissions, if any
    """
    if states is None:
        states = ["State_" + str(i) for i in range(model.n_states)]

    if emissions is None:
        emissions = create_emissions_name_list(model, hmm_type)

    print_startprob_table(model, states)
    print_transition_table(model, states)

    if hmm_type == "Multinomial":
        print_emission_table(model, states, emissions, hmm_type)
    elif hmm_type == "Gaussian":
        print_means_table(model, states, emissions)
        print_covars_table(model, states, emissions)
    elif hmm_type == "Heterogeneous":
        print_means_table(model, states, emissions[0])
        print_covars_table(model, states, emissions[0])
        print_emission_table(model, states, emissions[1], hmm_type)


def create_emissions_name_list(model, hmm_type="Multinomial"):
    """
    Helper method for the pretty print function. If the emissions
    are not given, it generates lists for the corresponding models.

    :param model: an HMM models
    :param str hmm_type: the type of the HMM; see pretty_print_hmm
    :return: a list or tuple of the generated emission labels
    """
    emissions = []
    if hmm_type == "Multinomial":
        for i in range(model.n_emissions):
            emissions.append(
                ["Emission_" + str(i) + str(j) for j in range(model.n_features[i])]
            )
    elif hmm_type == "Gaussian":
        emissions = ["Emission_" + str(i) for i in range(model.n_emissions)]
    elif hmm_type == "Heterogeneous":
        g_emissions = ["Gauss_" + str(i) for i in range(model.n_g_emissions)]
        d_emissions = []
        for i in range(model.n_d_emissions):
            d_emissions.append(
                ["Emission_" + str(i) + str(j) for j in range(model.n_d_features[i])]
            )
        emissions = (g_emissions, d_emissions)
    return emissions


def print_table(rows, header):
    """
    Helper method for the pretty print function. It prints the parameters
    as a nice table.

    :param list rows: the rows of the table
    :param list header: the header of the table
    """
    t = PrettyTable(header)
    for row in rows:
        t.add_row(row)
    print(t)


def print_startprob_table(model, states):
    """
    Helper method for the pretty print function. Prints the prior probabilities.

    :param model: an HMM models
    :param list states: the list of state names
    """
    print("Priors")
    rows = []
    for i, sp in enumerate(model.pi):
        rows.append("P({})={:.3f}".format(i, sp))
    print_table([rows], states)


def print_transition_table(model, states):
    """
    Helper method for the pretty print function. Prints the state transition probabilities.

    :param model: an HMM models
    :param list states: the list of state names
    """
    print("Transitions")
    rows = []
    for i, row in enumerate(model.A):
        rows.append(
            [states[i]]
            + ["P({}|{})={:.3f}".format(j, i, tp) for j, tp in enumerate(row)]
        )
    print_table(rows, ["_"] + states)


def print_emission_table(model, states, emissions, hmm_type):
    """
    Helper method for the pretty print function. Prints the emission probabilities.

    :param model: an HMM models
    :param list states: the list of state names
    :param list emissions: the list of emission names
    :param str hmm_type: the type of the HMM; see pretty_print_hmm
    """
    print("Emissions")
    n_emissions = model.n_emissions if hmm_type == "Multinomial" else model.n_d_emissions
    for e in range(n_emissions):
        rows = []
        for i, row in enumerate(model.B[e]):
            rows.append(
                [states[i]]
                + ["P({}|{})={:.3f}".format(j, i, ep) for j, ep in enumerate(row)]
            )
        print_table(rows, ["_"] + emissions[e])


def print_means_table(model, states, emissions):
    """
    Helper method for the pretty print function. Prints the means of the GaussianHMM.

    :param model: an HMM models
    :param list states: the list of state names
    :param list emissions: the list of emission names
    """
    print("Means")
    rows = []
    for i, row in enumerate(model.means):
        rows.append([states[i]] + ["{:.3f}".format(ep) for ep in row])
    print_table(rows, ["_"] + emissions)


def print_covars_table(model, states, emissions):
    """
    Helper method for the pretty print function. Prints the covariances of the GaussianHMM.

    :param model: an HMM models
    :param list states: the list of state names
    :param list emissions: the list of emission names
    """
    print("Covariances")
    for ns, state in enumerate(states):
        print(state)
        rows = []
        for i, row in enumerate(model.covars[ns]):
            rows.append([emissions[i]] + ["{:.3f}".format(ep) for ep in row])
        print_table(rows, ["_"] + emissions)
