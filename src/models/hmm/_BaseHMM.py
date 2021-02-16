# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Nov 20, 2019

@authors: semese, fmorenopino

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
 - HMM implementation by fmorenopino - https://github.com/fmorenopino/HMM_eb2
 - HMM implementation by anntzer - https://github.com/hmmlearn/

For theoretical bases see:
 - L. R. Rabiner, "A tutorial on hidden Markov models and selected applications
   in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

from multiprocessing import Pool

import numpy as np
from scipy.special import logsumexp

from src.utils.hmm_utils import (
    plot_log_likelihood_evolution,
    log_normalise,
    normalise,
    log_mask_zero,
    check_if_attributes_set,
)

#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class _BaseHMM(object):
    """
    Base class for Hidden Markov Models.
    """

    def __init__(
            self,
            n_states,
            params="st",
            init_params="st",
            init_type="random",
            pi_prior=1.0,
            A_prior=1.0,
            learn_rate=0,
            verbose=False,
    ):
        """
        Class initializer.

        :param int n_states: number of hidden states in the models
        :param str params: controls which parameters are updated in the
            training process.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
        :param str init_params: controls which parameters are initialised
            prior to training.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
        :param str init_type: name of the initialisation
            method to use for initialising the start,transition and emission
            matrices
        :param np.ndarray pi_prior: an array of shape (n_states, ) which
            gives the parameters of the Dirichlet prior distribution for 'pi'
        :param np.ndarray A_prior: array of shape (n_states, n_states)
            providing the parameters of the Dirichlet prior distribution for
            each row of the transition probabilities 'A'
        :param float learn_rate: a value from [0,1), controlling how much
            the past values of the models parameters count when computing the new
            models parameters during training. By default it's 0.
        :param bool verbose: flag to be set to True if per-iteration
            convergence reports should be printed
        """
        self.n_states = n_states
        self.params = params
        self.init_params = init_params
        self.init_type = init_type
        self.pi_prior = pi_prior
        self.A_prior = A_prior
        self.learn_rate = learn_rate
        self.verbose = verbose

    def __str__(self):
        """
            Function to allow directly printing the object.
        """
        return "Pi: " + str(self.pi) + "\nA:\n" + str(self.A)

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    # Solution to Problem 1 - compute P(O|models)
    def forward(self, observations, b_map=None):
        """
        Forward-Backward procedure is used to efficiently calculate the probability
        of the observations, given the models - P(O|models)

        alpha_t(x) = P(O1...Ot,qt=Sx|models) - The probability of state x and the
        observation up to time t, given the models.

        :param np.ndarray observations: a sequence of observations
        :param np.ndarray b_map:
        :return: The returned value is the log of the probability, i.e: the log likehood
            models, give the observation - logL(models|O).
        """
        if b_map is None:
            b_map = self._map_B(observations)
        alpha = self._calc_alpha(observations, b_map)
        return logsumexp(alpha[-1])

    def score(self, observation_sequences):
        """
        Compute the log probability under the models.

        :param list observation_sequences: list of observation sequences (ndarrays)
        :return: log-likelihood of all the observation sequences
        """
        log_likelihood = 0.0
        for observations in observation_sequences:
            log_likelihood += self.forward(observations)
        return log_likelihood

    def score_samples(self, observation_sequences, fwd_only=False):
        """
        Compute the posterior probability for each state in the models.

        :param list observation_sequences: a list of ndarrays containing the
            observation sequences of different lengths
        :param bool fwd_only: indicated whether both the forward and backward
            variables should be considered
        :return: list of arrays of shape (n_samples, n_states) containing the state-membership
            probabilities for each sample in the observation sequences
        """
        posteriors = []
        for observations in observation_sequences:
            b_map = self._map_B(observations)
            if fwd_only:
                posteriors.append(
                    self._calc_gamma(self._calc_alpha(observations, b_map))
                )
            else:
                posteriors.append(
                    self._calc_gamma(
                        self._calc_alpha(observations, b_map),
                        self._calc_beta(observations, b_map),
                    )
                )
        return posteriors

    # Solution to Problem 2 - finding the optimal state sequence associated with
    # the given observation sequence -> Viterbi, MAP
    def decode(self, observation_sequences, algorithm="viterbi"):
        """
        Find the best state sequence (path), given the models and an observation i.e: max(P(Q|O,models)).
        This method is usually used to predict the next state after training.

        :param list observation_sequences: a list of ndarrays containing the observation
            sequences of different lengths
        :param str algorithm: name of the decoder algorithm to use;
            must be one of "viterbi" or "map". Defaults to "viterbi".
        :return: log_likelihood (float) - log probability of the produced state sequence
            state_sequences (list) - list of arrays containing labels for each
            observation from observation_sequences obtained via the given
            decoder algorithm
        """
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {"viterbi": self._decode_viterbi, "map": self._decode_map}[algorithm]

        log_likelihood = 0.0
        state_sequences = []
        for observations in observation_sequences:
            log_likelihood_, state_sequence_ = decoder(observations)
            log_likelihood += log_likelihood_
            state_sequences.append(state_sequence_)

        return log_likelihood, state_sequences

    # Solution to Problem 3 - adjust the models parameters to maximise P(O,models)
    def train(
            self,
            observation_sequences,
            n_init=5,
            n_iter=100,
            thres=0.1,
            conv_iter=5,
            plot_log_likelihood=False,
            ignore_conv_crit=False,
            no_init=False,
            n_processes=None,
            print_every=1,
    ):
        """
        Updates the HMMs parameters given a new set of observed sequences.
        The observations can either be a single (1D) array of observed symbols, or when using
        a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
        time sample (multiple features).
        The models parameters are reinitialised 'n_init' times. For each initialisation the
        updated models parameters and the log likelihood is stored and the best models is selected
        at the end.

        :param list observation_sequences: a list of ndarrays containing the observation
            sequences of different lengths
        :param int n_init: number of initialisations to perform
        :param int n_iter: max number of iterations to run for each initialisation
        :param float thres: the threshold for the likelihood increase (convergence)
        :param int conv_iter: number of iterations before convergence is accepted
        :param bool ignore_conv_crit: flag to indicate whether to iterate until
            n_iter is reached or perform early stopping
        :param bool no_init: if True the models parameters are not initialised; only works if
            they have been initialised manually by the user
        :param int n_processes: number of processes to use if the training should
            be performed using parallelization
        :param bool plot_log_likelihood: parameter to activate plotting the evolution
            of the log-likelihood after each initialisation
        :param int print_every: if verbose is True, print progress info every
            'print_every' iterations
        :return: the trained models and the corresponding log-likelihood
        """

        # lists to temporarily save the new models parameters and corresponding log likelihoods
        new_models = []
        log_likelihoods = []
        log_likelihoods_plot = []
        for init in range(n_init):
            if self.verbose:
                print("Initialisation " + str(init + 1))

            new_model, log_likelihoods_init = self._train(
                observation_sequences,
                n_processes=n_processes,
                no_init=no_init,
                n_iter=n_iter,
                thres=thres,
                conv_iter=conv_iter,
                ignore_conv_crit=ignore_conv_crit,
                print_every=print_every
            )
            new_models.append(new_model)
            log_likelihoods.append(log_likelihoods_init[-1])
            log_likelihoods_plot.append(log_likelihoods_init)

        if plot_log_likelihood:
            plot_log_likelihood_evolution(log_likelihoods_plot, n_init != 1)

        # select best models (the one that had the largest log_likelihood) and update the models
        best_index = log_likelihoods.index(max(log_likelihoods))
        self._update_model(new_models[best_index])

        return self, max(log_likelihoods)

    def generate_sample_from_state(self, state):
        """
        Deriving classes should implement this method.
        Generates a random sample from a given state.

        :param int state: index of the component to condition on
        :return: X (array) - array of shape (n_features, ) containing a random sample
            from the emission distribution corresponding to a given state.
        """

    def sample(self, n_sequences=1, n_samples=1):
        """
        Generate samples from the models.

        :param int n_sequences: number of sequences to generate; by
                default it generates one sequence
        :param int n_samples: number of samples per sequence; if multiple
                sequences have to be generated, it is a list of the individual
                sequence lengths
        :return: samples (list) - a list containing one or n_sequences sample sequences
            state_sequences (list) - a list containing the state sequences that
                generated each sample sequence
        """
        samples = []
        state_sequences = []

        startprob_cdf = np.cumsum(self.pi)
        transmat_cdf = np.cumsum(self.A, axis=1)

        for ns in range(n_sequences):
            currstate = (startprob_cdf > np.random.rand()).argmax()
            state_sequence = [currstate]
            X = [self.generate_sample_from_state(currstate)]

            for t in range(n_samples - 1):
                currstate = (transmat_cdf[currstate] > np.random.rand()).argmax()
                state_sequence.append(currstate)
                X.append(self.generate_sample_from_state(currstate))
            samples.append(X)
            state_sequences.append(state_sequence)

        return samples, state_sequences

    def get_stationary_distribution(self):
        """
        Compute the stationary distribution of states. The stationary distribution is proportional to the
        left-eigenvector associated with the largest eigenvalue (i.e., 1) of the transition matrix.
        """
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init(self, X=None):
        """
        Initialises models parameters prior to fitting. If init_type if random,
        it samples from a Dirichlet distribution according to the given priors.
        Otherwise it initialises the starting probabilities and transition
        probabilities uniformly.
        :param list X: list of concatenated observations, only needed
            for the Gaussian models when random or K-means is used to initialise
            the means and covariances.
        """
        if self.init_type == "uniform":
            init = 1.0 / self.n_states
            if "s" in self.init_params:
                self.pi = np.full(self.n_states, init)

            if "t" in self.init_params:
                self.A = np.full((self.n_states, self.n_states), init)
        else:
            if "s" in self.init_params:
                self.pi = np.random.dirichlet(
                    alpha=self.pi_prior * np.ones(self.n_states), size=1
                )[0]

            if "t" in self.init_params:
                self.A = np.random.dirichlet(
                    alpha=self.A_prior * np.ones(self.n_states), size=self.n_states
                )

    def _decode_map(self, observations):
        """
        Find the best state sequence (path) using MAP.

        :param np.ndarray observations: array of shape (n_samples, n_features)
            containing the observation sequence
        :return: state_sequence (array) - the optimal path for the observation sequence
             log_likelihood (float) - the maximum probability for the entire sequence
        """
        posteriors = self.score_samples([observations])[0]
        log_likelihood = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return log_likelihood, state_sequence

    def _decode_viterbi(self, observations):
        """
        Find the best state sequence (path) using viterbi algorithm - a method
        of dynamic programming, very similar to the forward-backward algorithm,
        with the added step of maximisation and eventual backtracking.

        :param np.ndarray observations: array of shape (n_samples, n_features)
            containing the observation sequence
        :return: state_sequence (array) - the optimal path for the observation sequence
             log_likelihood (float) - the maximum probability for the entire sequence
        """
        n_samples = len(observations)

        # similar to the forward-backward algorithm, we need to make sure that
        # we're using fresh data for the given observations
        B_map = self._map_B(observations)

        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # delta[t][i] = max(P[q1..qt=i,O1...Ot|models] - the path ending in Si and
        # until time t, that generates the highest probability.
        delta = np.zeros((n_samples, self.n_states))

        # init
        for x in range(self.n_states):
            delta[0][x] = log_pi[x] + log_B_map[x][0]

        # induction
        work_buffer = np.empty(self.n_states)
        for t in range(1, n_samples):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[j] = log_A[j][i] + delta[t - 1][j]
                delta[t][i] = np.amax(work_buffer) + log_B_map[i][t]

        # Observation traceback
        state_sequence = np.empty(n_samples, dtype=np.int32)
        state_sequence[n_samples - 1] = where_from = np.argmax(delta[n_samples - 1])
        log_likelihood = delta[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                work_buffer[i] = delta[t, i] + log_A[i, where_from]
            state_sequence[t] = where_from = np.argmax(work_buffer)

        return log_likelihood, state_sequence

    def _calc_alpha(self, observations, b_map):
        """
        Calculates 'alpha' the forward variable given an observation sequence.

        :param np.ndarray observations: array of shape (n_samples, n_features)
            containing the observation samples
        :param np.ndarray b_map: the observations' mass/density Bj(Ot) to Bj(t)
        :return: alpha (array) - array of shape (n_samples, n_states) containing
            the forward variables
        """
        n_samples = len(observations)

        # The alpha variable is a np array indexed by time, then state (TxN).
        # alpha[t][i] = the probability of being in state 'i' after observing the
        # first t symbols.
        alpha = np.zeros((n_samples, self.n_states))
        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_b_map = log_mask_zero(b_map)

        # init stage - alpha_1(i) = pi(i)b_i(o_1)
        for i in range(self.n_states):
            alpha[0][i] = log_pi[i] + log_b_map[i][0]

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                for i in range(self.n_states):
                    work_buffer[i] = alpha[t - 1][i] + log_A[i][j]
                alpha[t][j] = logsumexp(work_buffer) + log_b_map[j][t]

        return alpha

    def _calc_beta(self, observations, b_map):
        """
        Calculates 'beta' the backward variable for each observation sequence.

        :param np.ndarray observations: array of shape (n_samples, n_features)
                containing the observation samples
        :param np.ndarray b_map: the observations' mass/density Bj(Ot) to Bj(t)
        :return: beta (array) - array of shape (n_samples, n_states) containing
            the backward variables
        """
        n_samples = len(observations)

        # The beta variable is a ndarray indexed by time, then state (TxN).
        # beta[t][i] = the probability of being in state 'i' and then observing the
        # symbols from t+1 to the end (T).
        beta = np.zeros((n_samples, self.n_states))

        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(b_map)

        # init stage
        for i in range(self.n_states):
            beta[len(observations) - 1][i] = 0.0

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[j] = log_A[i][j] + log_B_map[j][t + 1] + beta[t + 1][j]
                    beta[t][i] = logsumexp(work_buffer)

        return beta

    def _calc_xi(
            self, observations, b_map=None, alpha=None, beta=None
    ):
        """
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        :param np.ndarray observations: array of shape (n_samples, n_features)
            containing the observation samples
        :param np.ndarray b_map: the observations' mass/density Bj(Ot) to Bj(t)
        :param np.ndarray alpha: array of shape (n_samples, n_states) containing the forward variables
        :param np.ndarray beta: array of shape (n_samples, n_states) containing the backward variables
        :return: xi (array) - array of shape (n_samples, n_states, n_states) containing
            the a joint probability from the 'alpha' and 'beta' variables.
        """
        if b_map is None:
            b_map = self._map_B(observations)
        if alpha is None:
            alpha = self._calc_alpha(observations, b_map)
        if beta is None:
            beta = self._calc_beta(observations, b_map)

        n_samples = len(observations)

        # The xi variable is a np array indexed by time, state, and state (TxNxN).
        # xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        # time 't+1' given the entire observation sequence.
        log_xi_sum = np.zeros((self.n_states, self.n_states))
        work_buffer = np.full((self.n_states, self.n_states), -np.inf)

        # compute the logarithm of the parameters
        log_A = log_mask_zero(self.A)
        log_b_map = log_mask_zero(b_map)
        logprob = logsumexp(alpha[n_samples - 1])

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[i, j] = (
                            alpha[t][i]
                            + log_A[i][j]
                            + log_b_map[j][t + 1]
                            + beta[t + 1][j]
                            - logprob
                    )

            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi_sum[i][j] = np.logaddexp(log_xi_sum[i][j], work_buffer[i][j])

        return log_xi_sum

    def _calc_gamma(self, alpha, beta=None):
        """
        Calculates 'gamma' from 'alpha' and 'beta'.

        :param np.ndarray alpha: array of shape (n_samples, n_states) containing the forward variables
        :param np.ndarray beta: array of shape (n_samples, n_states) containing the backward variables
        :return: gamma (array) - array of shape (n_samples, n_states), the posteriors
        """
        log_gamma = alpha + beta if beta is not None else alpha
        log_normalise(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    # Methods used by self.train()
    def _train(
            self,
            observation_sequences,
            n_iter=100,
            thres=0.1,
            conv_iter=5,
            ignore_conv_crit=False,
            no_init=False,
            print_every=1,
            n_processes=None,
    ):
        """
        Training is repeated 'n_iter' times, or until log likelihood of the models
        increases by less than a threshold.

        :param list observation_sequences: a list of ndarrays containing the observation
            sequences of different lengths
        :param int n_iter: max number of iterations to run for each initialisation
        :param float thres: the threshold for the likelihood increase (convergence)
        :param int conv_iter: number of iterations before convergence is accepted
        :param bool ignore_conv_crit: flag to indicate whether to iterate until
            n_iter is reached or perform early stopping
        :param bool no_init: if True the models parameters are not initialised; only works if
            they have been initialised manually by the user
        :param int print_every: if verbose is True, print progress info every
            'print_every' iterations
        :param int n_processes: number of processes to use if the training should
            be performed using parallelization
        :return: new_model (dictionary) - containing the updated models parameters
            log_likelihood_iter (list, optional) - the log-likelihood values from each iteration - returned if return_log_likelihoods = True
            curr_log_likelihood (float) - the accumulated log-likelihood for all the observations  - returned if return_log_likelihoods = False
        """
        if not no_init:
            if self.init_type in ("kmeans", "random"):
                self._init(X=observation_sequences)
            else:
                self._init()
        else:
            check_if_attributes_set(self)

        log_likelihood_iter = []
        old_log_likelihood = np.nan
        curr_log_likelihood = np.nan
        counter = 0
        new_model = None

        for it in range(n_iter):

            # if train without multiprocessing
            if n_processes is None:
                stats, curr_log_likelihood = self._compute_intermediate_values(
                    observation_sequences
                )
            else:
                # split up observa sequences between the processes
                n_splits = int(np.ceil(len(observation_sequences) / n_processes))
                split_list = [
                    sl
                    for sl in list(
                        (
                            observation_sequences[
                            i * n_splits: i * n_splits + n_splits
                            ]
                            for i in range(n_processes)
                        )
                    )
                    if sl
                ]
                # create pool of processes
                p = Pool(processes=n_processes)
                stats_list = p.map(
                    self._compute_intermediate_values,
                    [split_i for split_i in split_list],
                )
                p.close()
                stats, curr_log_likelihood = self._sum_up_sufficient_statistics(
                    stats_list
                )

            # perform the M-step to update the models parameters
            new_model = self._M_step(stats)
            self._update_model(new_model)

            if self.verbose and it % print_every == 0:
                print(
                    "iter: {}, logL = {:.3f}, delta = {:.3f}".format(
                        it,
                        curr_log_likelihood,
                        (curr_log_likelihood - old_log_likelihood),
                    )
                )

            if not ignore_conv_crit:
                if (
                        abs(curr_log_likelihood - old_log_likelihood)
                        / abs(old_log_likelihood)
                        <= thres
                ):
                    counter += 1
                    if counter == conv_iter:
                        # converged
                        if self.verbose:
                            print(
                                "Converged -> iter: {}, logL = {:.3f}".format(
                                    it, curr_log_likelihood
                                )
                            )
                        break
                else:
                    counter = 0

            log_likelihood_iter.append(curr_log_likelihood)
            old_log_likelihood = curr_log_likelihood

        if counter < conv_iter and self.verbose:
            # max_iter reached
            print(
                "Maximum number of iterations reached. logL = {:.3f}".format(
                    curr_log_likelihood
                )
            )

        return new_model, log_likelihood_iter

    def _compute_intermediate_values(self, observation_sequences):
        """
        Calculates the various intermediate values for the Baum-Welch on a list
        of observation sequences.

        :param list observation_sequences: a list of ndarrays/lists containing
            the observation sequences. Each sequence can be the same or of
            different lengths
        :return: stats (dictionary) - dictionary of sufficient statistics required
            for the M-step
        """
        stats = self._initialise_sufficient_statistics()
        curr_log_likelihood = 0

        for observations in observation_sequences:
            b_map = self._map_B(observations)

            # calculate the log likelihood of the previous models
            # we compute the P(O|models) for the set of old parameters
            log_likelihood = self.forward(observations, b_map)
            curr_log_likelihood += log_likelihood

            # do the E-step of the Baum-Welch algorithm
            observations_stats = self._E_step(observations, b_map)

            # accumulate stats
            self._accumulate_sufficient_statistics(
                stats, observations_stats, observations
            )

        return stats, curr_log_likelihood

    def _E_step(self, observations, b_map):
        """
        Calculates required statistics of the current models, as part
        of the Baum-Welch 'E' step. Deriving classes should override (extend) this method to include
        any additional computations their models requires.

        :param np.ndarray observations: array of shape (n_samples, n_features)
            containing the observation samples
        :param np.ndarray b_map: the observations' mass/density Bj(Ot) to Bj(t)
        :return: observations_stats (dictionary) - containing required statistics
        """

        # compute the parameters for the observations
        observations_stats = {
            "alpha": self._calc_alpha(observations, b_map),
            "beta": self._calc_beta(observations, b_map),
        }

        observations_stats["xi"] = self._calc_xi(
            observations,
            b_map=b_map,
            alpha=observations_stats["alpha"],
            beta=observations_stats["beta"],
        )
        observations_stats["gamma"] = self._calc_gamma(
            observations_stats["alpha"], observations_stats["beta"]
        )
        return observations_stats

    def _M_step(self, stats):
        """
        Performs the 'M' step of the Baum-Welch algorithm.
        Deriving classes should override (extend) this method to include
        any additional computations their models requires.

        :param dict stats: containing the accumulated statistics
        :return: new_model (dictionary) - containing the updated models parameters
        """
        new_model = {}

        if "s" in self.params:
            pi_ = np.maximum(self.pi_prior - 1 + stats["pi"], 0)
            new_model["pi"] = np.where(self.pi == 0, 0, pi_)
            normalise(new_model["pi"])

        if "t" in self.params:
            A_ = np.maximum(self.A_prior - 1 + stats["A"], 0)
            new_model["A"] = np.where(self.A == 0, 0, A_)
            normalise(new_model["A"], axis=1)

        return new_model

    def _update_model(self, new_model):
        """
        Replaces the current models parameters with the new ones.

        :param dict new_model: contains the new models parameters
        :return:
        """
        if "s" in self.params:
            self.pi = (1 - self.learn_rate) * new_model[
                "pi"
            ] + self.learn_rate * self.pi

        if "t" in self.params:
            self.A = (1 - self.learn_rate) * new_model["A"] + self.learn_rate * self.A

    def _initialise_sufficient_statistics(self):
        """
        Initialises sufficient statistics required for M-step.
        """
        stats = {
            "nobs": 0,
            "pi": np.zeros(self.n_states),
            "A": np.zeros((self.n_states, self.n_states)),
        }
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, observations_stats, observations
    ):
        """
        Updates sufficient statistics from a given sample.

        :param dict stats: containing the sufficient statistics for all
            observation sequences
        :param dict observations_stats: containing the sufficient statistic for one sample
        """
        stats["nobs"] += 1
        if "s" in self.params:
            stats["pi"] += observations_stats["gamma"][0]

        if "t" in self.params:
            with np.errstate(under="ignore"):
                stats["A"] += np.exp(observations_stats["xi"])

    def _sum_up_sufficient_statistics(self, stats_list):
        """
        Updates sufficient statistics from a given sample.

        :param list stats_list: list containing the sufficient statistics from the
            different processes
        :return: stats_all (dictionary) - dictionary of sufficient statistics
        """
        stats_all = self._initialise_sufficient_statistics()
        logL_all = 0
        for (stat_i, logL_i) in stats_list:
            logL_all += logL_i
            for stat in stat_i.keys():
                if isinstance(stat_i[stat], dict):
                    for i in range(len(stats_all[stat]["numer"])):
                        stats_all[stat]["numer"][i] += stat_i[stat]["numer"][i]
                        stats_all[stat]["denom"][i] += stat_i[stat]["denom"][i]
                else:
                    stats_all[stat] += stat_i[stat]
        return stats_all, logL_all

    # Methods that have to be implemented in the deriving classes
    def _map_B(self, observations):
        """
        Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t).
        This method has no explicit return value, but it expects that 'b_map'
         is internally computed as mentioned above.
        'b_map' is an (TxN) numpy array.
        The purpose of this method is to create a common parameter that will
        conform both to the discrete case where PMFs are used, and the continuous
        case where PDFs are used.
        For the continuous case, since PDFs of vectors could be computationally
        expensive (Matrix multiplications), this method also serves as a caching
        mechanism to significantly increase performance.

        :param np.ndarray observations: array of shape (n_samples, n_features)
            containing the observation samples
        """
        raise NotImplementedError(
            "a mapping function for B(observable probabilities) must be implemented"
        )
