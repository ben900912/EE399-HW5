# EE-399-HW5
``Author: Ben Li``
``Date: 5/14/2023``
``Course: SP 2023 EE399``

## Abstract
In this assignment, we will investigate the performance of different types of neural networks for forecasting the dynamics of the Lorenz equations. 

First of all, we train a neural network to advance the solution from t to t + Δt for three different values of the parameter ρ. 

Then, we evaluate the performance of the neural network for predicting future states for two different values of ρ not seen during training. Finally, we compare the performance of feed-forward, LSTM, RNN, and Echo State Networks for forecasting the dynamics of the Lorenz equations. Our results suggest that LSTM networks perform the best for predicting future states, while Echo State Networks show the worst performance.

## Introduction and Overview
In the first part of the assignment, we train a neural network to advance the solution from t to t + Δt for three different values of the parameter ρ. We then evaluate the performance of the neural network for predicting future states for two different values of ρ not seen during training.

In the second part of the project, we compare the performance of different types of neural networks for forecasting the dynamics of the Lorenz equations. We consider feed-forward neural networks, LSTM networks, RNNs, and Echo State Networks, and compare their performance in terms of accuracy and computational efficiency.

Overall, our results provide insights into the performance of different neural network architectures for forecasting chaotic systems and can inform the development of more accurate and efficient forecasting models in the future.
## Theoretical Background
The Lorenz equations are a set of three nonlinear differential equations that describe the dynamics of a simple model of atmospheric convection. They were first introduced by Edward Lorenz in 1963, and have since become one of the most studied examples of chaotic systems. The equations are given by:

$$dx/dt = σ(y - x)$$

$$dy/dt = x(ρ - z) - y$$

$$dz/dt = xy - βz $$

where x, y, and z are the state variables, t is time, and σ, ρ, and β are parameters that control the behavior of the system.

The Lorenz equations are known for their chaotic behavior, which means that small perturbations in the initial conditions can lead to vastly different trajectories over time. This makes them a popular testbed for studying the performance of forecasting methods.

Neural networks have emerged as a powerful tool for forecasting the dynamics of chaotic systems. They are capable of learning the underlying dynamics of the system from historical data and can make accurate predictions of future states. Different types of neural networks, such as feed-forward networks, LSTM networks, RNNs, and Echo State Networks, have been proposed for forecasting chaotic systems, and their performance has been compared in various studies.
## Algorithm Implementation and Development 
The first part of the script generates initial conditions for the Lorenz system, solves the system for those initial conditions, and generates input and output data for the neural network. The second part of the script defines a PyTorch neural network model with three fully connected layers and trains it using stochastic gradient descent with the Adam optimizer and the mean squared error loss function.

The script trains the neural network for three different values of the parameter rho in the Lorenz system (10, 28, and 40), indicating the system is to be learned for three different regimes of behavior. This can be useful for studying the dynamics of the Lorenz system under different conditions
## Computational Results
``
## Summary and Conclusions
