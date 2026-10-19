# Somesul Mic Streamflow Uncertainty

A PyTorch model designed to predict river discharge and estimate heteroscedastic uncertainty for the Somesul Mic river basin.

Instead of predicting a single deterministic value, the network estimates both the expected discharge (mu) and the uncertainty (sigma).

## Results

![Model Predictions](somesul_mic_flood_predictions.png)

The uncertainty band (+-2 sigma) expands as precipitation increases, reflecting higher discharge variance during heavy rainfall events.

## Methodology

- Architecture: Multi-layer perceptron with two outputs (mu and raw sigma)
- Positivity constraint: Softplus activation to ensure sigma is strictly positive
- Loss function: Gaussian Negative Log-Likelihood (nn.GaussianNLLLoss)
- Risk calculation: torch.distributions.Normal to compute threshold exceedance probabilities

## Flood Risk Evaluation

Evaluation for an 80 mm rainfall event:

- Estimated Mean Discharge: 108.02 m3/s
- Associated Uncertainty: 12.53 m3/s
- Critical Levee Threshold: 110.0 m3/s
- Probability of Overtopping: 43.72%

Although the mean predicted flow is below the levee threshold, the model captures a 43.72% chance of flooding due to variance in runoff.

## Dependencies

- python 3
- torch
- numpy
- matplotlib
