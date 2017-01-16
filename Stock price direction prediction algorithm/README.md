#### Prediction of stock price direction near support levels using ANNs is the official project name.

First, one has to understand what support and resistance levels are. Then, appropiate samples, including features, are generated using stock data from Yahoo through Matlab. 

An artificial neural network is then trained to determine if the price of a stock will go up or down relative to the support level. 

Due to hardware and time constraints, many of the ANN configuration parameters were not uploaded to github. That is, here, a subset of the parameters were used:
* Only 10 neurons in the hidden layer,
* One training algorithm (Resilient Backprop),
* One hidden layer
* Feature vector of 2 features (out of 19 features I developed)

I will add more information to this description at some point.
