Using Keras I am building a Recurrent Neural Network (RNN) to predict the upward and downward trends that exist in Google Inc stock price. Here I will train the RNN model with 4 Years of the stock price from the beginning of 2012 till the end of 2016. Later will try to predict the stock price from 2017 – 2018.

At the end of the prediction I will use Bokeh to create an interactive visualization chart to display information regarding predicted stock price and existing stock price.

I have used four LTSM input layers with 50 each unit that is the neuron in the current neural network and one Dense output layer which is connected to the previous LSTM layer which has units equal to 1 since the output we are expecting will be only one set.

I have assigned epoch value equal to 100 since I could find more convergence than 25 or 50 epochs values.