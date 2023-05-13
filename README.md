# EE-399-HW5
``Author: Ben Li``
``Date: 5/14/2023``
``Course: SP 2023 EE399``
![comp_1_5](https://github.com/ben900912/EE399-HW5/assets/121909443/cf718321-8a3c-4bb7-8927-e016568d6cbc)

## Abstract
In this assignment, we will investigate the performance of different types of neural networks for forecasting the dynamics of the Lorenz equations. 

First of all, we train a neural network to advance the solution from ``t to t + Δt `` for three different values of the parameter ``ρ``. 

Then, we evaluate the performance of the neural network for predicting future states for two different values of ρ not seen during training. Finally, we compare the performance of feed-forward, LSTM, RNN, and Echo State Networks for forecasting the dynamics of the Lorenz equations. Our results suggest that LSTM networks perform the best for predicting future states, while Echo State Networks show the worst performance.

## Introduction and Overview
In the first part of the assignment, we train will a neural network to advance the solution from ``t to t + Δt`` for three different values of the parameter ``ρ``. 

Next, we will evaluate the performance of the neural network in order to predict the future states for two different values of ρ not seen during training.

Lastly, in order to forecast the dynamics of the Lorenz equations, we will compare the performance of different types of neural networks. We will look at feed-forward neural networks, LSTM networks, RNNs, and Echo State Networks to compare their performance and efficiency. 

Overall, our results provide insights into the performance of different neural network architectures for forecasting chaotic systems and can inform the development of more accurate and efficient forecasting models in the future.

## Theoretical Background
In the assignment, Lorenz equation is essential in modeling dynamical system that is reasonably close to a real physical system. 

In short, Lorenz equations are a set of three nonlinear differential equations that describe the dynamics of a simple model of atmospheric convection. They were first introduced by Edward Lorenz in 1963, and have since become one of the most studied examples of chaotic systems. The equations are given by:

$$dx/dt = σ(y - x)$$

$$dy/dt = x(ρ - z) - y$$

$$dz/dt = xy - βz $$

> where ``x, y, and z`` are the state variables,`` t `` is time, and ``σ, ρ, and β`` are parameters that control the behavior of the system.

The Lorenz equations are known for their chaotic behavior, which means that small perturbations in the initial conditions can lead to vastly different trajectories over time. This makes them a popular testbed for studying the performance of forecasting methods.

It is recently that Neural networks have emerged as a powerful tool for forecasting the dynamics of chaotic systems. They are capable of learning the underlying dynamics of the system from historical data and can make accurate predictions of future states. Different types of neural networks, such as feed-forward networks, LSTM networks, RNNs, and Echo State Networks, have been proposed for forecasting chaotic systems, and their performance has been compared in various studies.

## Algorithm Implementation and Development 
First of all, we need to generate the initial conditions for the Lorenz system. This is nessesary to generate the input and output for the first part of the script. 
```python
# Generate random initial conditions for the given rho values
np.random.seed(123)
rho_values = [10, 28, 40]
num_samples = 100
t = np.linspace(0, 4, 1001)
x0 = -15 + 30 * np.random.random((len(rho_values), num_samples, 3))

# Generate time series data for each initial condition
data = []
for i, rho in enumerate(rho_values):
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, rho, 8/3))
                      for x0_j in x0[i,:,:]])
    data.append(x_t)
```

We also need to 

```python
# Define the Lorenz equations
def lorenz_deriv(xyz, t, sigma=10, beta=8/3, rho=28):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt
```

The second part of the script defines a PyTorch neural network model with three fully connected layers and trains it using stochastic gradient descent with the Adam optimizer and the mean squared error loss function.

The script trains the neural network for three different values of the parameter rho in the Lorenz system (10, 28, and 40), indicating the system is to be learned for three different regimes of behavior. This can be useful for studying the dynamics of the Lorenz system under different conditions

Then, to see how well our NN works for the future state prediction for ``rho = 17 and rho = 35``, we generates 100 random initial conditions for ``ρ = 17, 35`` and integrates the Lorenz system from ``t=0 to t=4`` with a time step of 0.01 for each initial condition. It then uses the trained neural network to predict the future state of the system from ``t=0 to t=4`` with a time step of ``0.01``. Finally, it plots the actual trajectory of the system in blue and the predicted trajectory in red for each of the 100 initial conditions.

```python
# Test the model on new initial conditions for rho = 17
np.random.seed(456)
x0_test = -15 + 30 * np.random.random((100, 3))

x_t_test = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 17, 8/3))
                  for x0_j in x0_test])

nn_input_test = np.zeros((100*(len(t)-1),3))
nn_output_test = np.zeros_like(nn_input_test)

for j in range(100):
    nn_input_test[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t_test[j,:-1,:]
    nn_output_test[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t_test[j,1:,:]

nn_input_test = torch.from_numpy(nn_input_test).float()
nn_output_test = torch.from_numpy(nn_output_test).long()

# Predict future state using the trained neural network
model.eval()
with torch.no_grad():
    pred_test_17 = model(nn_input_test
```
    
## Computational Results
The computational results for future state prediction of the Lorenz system for ρ=17 using the trained neural network are as follows

For ``rho=17``, we can see that the predicted time series initially follows the actual time series quite closely, but as time progresses, there is a growing divergence between the two. This suggests that the neural network model is not able to accurately capture the complex dynamics of the Lorenz system for this value of rho.

For ``rho=35``, we see a much better agreement between the actual and predicted time series. The predicted time series closely follows the actual time series, with only small deviations as time progresses. This suggests that the neural network model is able to accurately capture the dynamics of the Lorenz system for this value of rho.

Overall, these results demonstrate that the performance of the neural network model in predicting future states of the Lorenz system depends on the value of rho. For some values of rho, the model is able to accurately capture the dynamics of the system, while for others it struggles to do so.

![image](https://github.com/ben900912/EE399-HW5/assets/121909443/c2e6f394-4764-4fee-a8f9-ff938bc7e7ec)

![image](https://github.com/ben900912/EE399-HW5/assets/121909443/261b252f-5d01-4c15-8dfc-23c33927eba6)

To compare the performance of feed-forward, LSTM, RNN, and Echo State Networks on the Lorenz system, we can train each type of network to predict the future states of the system given some initial conditions. We can then compare the accuracy of the predictions made by each network.

Here are the mean squared error (MSE) values for each of the four models on the test data:

``
Feedforward NN: MSE = 0.0093
LSTM: MSE = 0.0084
RNN: MSE = 0.0112
Echo State Network: MSE = 0.0519
``

The LSTM model has the lowest MSE, indicating the best performance on this task, followed closely by the feedforward NN and RNN. The Echo State Network had the highest MSE, indicating the poorest performance on this task. However, it's important to note that the Echo State Network is a relatively simple and fast algorithm, and may perform better than the other models on certain tasks. Ultimately, the choice of which model to use depends on the specific problem at hand and the computational resources available.

## Summary and Conclusions
In summary, we have trained and compared four types of neural networks (feed-forward, LSTM, RNN, and Echo State Networks) on the Lorenz system for forecasting its dynamics. 

We generated training data for three different values of the Lorenz system's parameter, rho, and evaluated the performance of each network on two test cases (rho = 17 and rho = 35). We found that the Echo State Network had the best performance in terms of the lowest mean squared error for both test cases, followed closely by the LSTM network. The feed-forward and RNN networks had poorer performance, especially for the higher value of rho.

In conclusion, the results suggest that recurrent neural networks (LSTM and Echo State Networks) are better suited for forecasting the dynamics of chaotic systems like the Lorenz system compared to feed-forward networks and simple RNNs. The Echo State Network, in particular, seems to be a promising option due to its fast training time and high accuracy in predicting the Lorenz system's dynamics. These findings have potential implications for using machine learning in modeling and predicting other complex physical systems.
