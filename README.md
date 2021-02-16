# Mood prediction from mobile sensed data

Code for "Predicting Emotional State Using Behavioural Markers Derived from Passively Sensed Data: a Data-Driven Approach" by Emese Sukei, Agnes Norbury, M.Mercedes Perez-Rodriguez, Pablo M. Olmos, Antonio Artés Rodríguez. 

In this project, a machine learning-based approach for emotional state (mood) prediction was implemented. The dataset for this project consisted of passively-collected data from mobile phones and wearable devices, as well as self-reported emotions of a cohort of N = 943 individuals (outpatients recruited from community clinics). All patients had at least 30 days worth of naturally-occurring behaviour observations, including information about physical activity, geolocation, sleep, and smartphone app usage.

The time series data at hand was heterogeneous, regularly sampled, but frequently missing. Therefore, first probabilistic latent variable models were used for data averaging and feature extraction: Mixture Model (MM) and Hidden Markov Model (HMM). The extracted features were then combined with a classifier to provide emotional state predictions. Three different settings were analysed for the classifiers’ input features: using the imputed raw data, using the MM/HMM posterior probabilities instead of the raw input features and using the raw inputs concatenated with the MM/HMM posterior probabilities. 

A variety of classical machine learning methods and recurrent neural networks were compared in the study. Finally, a personalised Bayesian model was proposed to improve the performance, which considers the individual differences in the data by applying a different classifier bias term for each patient. In this repository examples are given for using a Balanced Random Forest Classifier (BRFC) with 7-days of concatenated data as input feature, a Recurrent Neural Network (RNN) with Gated Recurrent Unit (GRU) cells using 30-day long sequences, and the Hierarchical Bayesian Logistic Regression (HBLR) using 1-day of observations as input feature.

More details on the experiments can be found in the above mentioned paper. 

