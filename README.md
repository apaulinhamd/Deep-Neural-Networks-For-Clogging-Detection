# Use Of Deep Neural Networks For Clogging Detection In The Submerged Entry Nozzle Of The Continuous Casting
 Use Of Deep Neural Networks For Clogging Detection In The Submerged Entry Nozzle Of The Continuous Castinge


This repository aims to share the codes generated for detecting occurrences of clogging. This research employs six months of historical data from measurements taken in a continuous casting steelmaking process, with variables sampled at a rate of one sample per second and collected from two distributors operating on a mold and a tundish. To do so, we are using four input variables obtained from a real continuous casting process.

Data preprocessing to handle noise, outliers, and missing samples is not within the scope of this project.

It is important to highlight that, due to data confidentiality, we do not have authorization to disclose the database used in this project.

# Project Objective:
The objective of this project is to compare the overall performance of classifiers using LSTM (lstm_github.py) and its variant, ConvLSTM (convlstm_github.py). Additionally, it evaluates the performance of a hybrid approach, which combines CNN1D with LSTM, called CNN-LSTM (cnnlstm_github.py). To evaluate the efficiency of the proposed models, MLP models (mlp_github.py) were used as a baseline for comparison.

To make comparative analyses, preliminary tests were conducted with different constructive topologies for each network, defined by trial and error. The parameters considered were the number of layers, the number of cells in the LSTM layers, the quantity and size of filters in the convolutional layers, the quantity of units in the FC layers, and the use of MaxPooling and dropout to evaluate how each of the parameters could influence the network's performance.

Different configurations of MLP networks were tested with one or two hidden layers with sigmoid activation functions, varying from 16 to 4096 neurons. Based on the preliminary test results, a dropout of 0.5 was applied in the second MLP layer, while other network hyperparameters were kept as default.

# Formation of input matrices:
The input data is organized into an input matrix grouped by means of a sliding window, with 120 overlapping samples with a 1-sample per timestep step. Each matrix is labeled with one of the corresponding classes of one of the events (clogging or normal operation) and is formed based on four selected variables with delayed samples in steps of 120 seconds. The "series_to_supervised" function is used to perform this process.

# i) Data Balancing:
The training dataset was balanced using the Undersampling technique to equalize the number of observations between the clogging and normal operation classes. This technique was not applied to the validation and test datasets, maintaining the original proportion of 93% and 7% for the clogging and normal operation classes, respectively.

# ii) Data Set Split
80% of the data will be used for training, 10% will be used for validation, and the remaining 10% will be reserved for testing.

# iii) Data Normalization
Using the mean and standard deviation of each variable, calculated from the training dataset, the data is standardized using the z-score.


# iv) Input structure:

The input structure refers to the organization of the data that will be provided as input to neural network models. In the context of the text, input structures were discussed for different types of models, such as LSTM, convolutional layers, and ConvLSTM. For each type of model, the format of the tensor representing the input was specified, taking into account factors such as the number of samples, the size of the time window, and the number of input attributes. The choice of the appropriate input structure can significantly influence the model's performance on the task at hand.

- The input to the LSTM is a 3D tensor in the format N ×120×4, where N represents the number of samples, 120 is the size of the time window, and 4 is the number of input attributes of the model.

- For the input to the convolutional layer (used in the CNN-LSTM), it is necessary to reshape it into a 4D tensor in the format N ×4×30×1 by dividing each 120-second window into 4 subsequences of 30 seconds.

- The input to the ConvLSTM is a 5D tensor in the format N×4×1×30×4, where N represents the number of samples, 4 is the number of subsequences in which each instance is divided, 1 is the data dimension, 30 is the number of time intervals in each subsequence, and 4 is the number of input attributes.


# Biased MLP:
To demonstrate the superior generalization capacity of ConvLSTM models compared to the baseline, we performed biased training on MLP models. For this purpose, the test set was also used during the training phase. This step is commented in the MLP_github.py code.

# Post-processing heuristic:

After training the networks, a type of post-processing heuristic was also tested in order to reduce false positive and false negative classification rates. Based on the training data, it was found that clogging occurrences can last from 45 seconds up to 45 minutes, depending on the control action performed by the operator, steel type, and other process conditions. Using this information, the model's response was evaluated in 40-second windows, i.e., blocks containing 40 matrices each.
In each window, the number of samples classified as clogging was calculated relative to the total window size, called ClgMed. If the clogging occurrence is below the 30% threshold (LimMin), the system will classify this window as normal operation. On the other hand, if the clogging occurrence is above the 80% threshold (LimMax), the system will automatically classify the window as clogging.

For cases where the window presents a clogging occurrence between the minimum and maximum limits, the algorithm evaluates the classification of the previous window. If the previously classified window is labeled as clogging, the current window will also be labeled as obstruction. Otherwise, the window will be classified as normal operation.

This procedure is in the convlstm_github_heuristica.py code.
