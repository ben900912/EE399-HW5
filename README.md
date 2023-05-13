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

Then, to see how well our NN works for the future state prediction for rho = 17 and rho = 35, we generates 100 random initial conditions for ρ = 17, 35 and integrates the Lorenz system from t=0 to t=4 with a time step of 0.01 for each initial condition. It then uses the trained neural network to predict the future state of the system from t=0 to t=4 with a time step of 0.01. Finally, it plots the actual trajectory of the system in blue and the predicted trajectory in red for each of the 100 initial conditions.
## Computational Results

The computational results for future state prediction of the Lorenz system for ρ=17 using the trained neural network are as follows
For rho=17, we can see that the predicted time series initially follows the actual time series quite closely, but as time progresses, there is a growing divergence between the two. This suggests that the neural network model is not able to accurately capture the complex dynamics of the Lorenz system for this value of rho.

For rho=35, we see a much better agreement between the actual and predicted time series. The predicted time series closely follows the actual time series, with only small deviations as time progresses. This suggests that the neural network model is able to accurately capture the dynamics of the Lorenz system for this value of rho.

Overall, these results demonstrate that the performance of the neural network model in predicting future states of the Lorenz system depends on the value of rho. For some values of rho, the model is able to accurately capture the dynamics of the system, while for others it struggles to do so.

![image](https://github.com/ben900912/EE399-HW5/assets/121909443/c2e6f394-4764-4fee-a8f9-ff938bc7e7ec)

![image](https://github.com/ben900912/EE399-HW5/assets/121909443/261b252f-5d01-4c15-8dfc-23c33927eba6)

## Summary and Conclusions
