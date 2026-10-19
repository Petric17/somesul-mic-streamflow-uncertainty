import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from torch.distributions import Normal

X = torch.linspace(5, 85, 400)
X = X.unsqueeze(1)

print(X.shape)

#Hydrological params

baseflow = 12.0
runoff_coefficient = 1.2

# Mean discharge (mu)

mean_discharge = baseflow + runoff_coefficient * X

#Uncertainty (sigma)

sigma = 1 + 0.15 * X

# River discharge with gaussian nois 

Y = mean_discharge + sigma * torch.randn_like(X)

#Model Building

class StreamFlowUncertaintyModle(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(1, 64)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(64, 2)

    def forward(self,x):
        x = self.hidden(x)
        x = self.activation(x)
        out = self.output_layer(x)
        mu = out[:, 0:1]
        sigma_raw = out[:, 1:2]
        sigma = torch.nn.functional.softplus(sigma_raw) + 1e-4
        return mu, sigma 
              


    
model = StreamFlowUncertaintyModle()

#Loss Function
criterion = nn.GaussianNLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 800

for epoch in range(epochs):
    optimizer.zero_grad()
    mu_pred, sigma_pred = model(X)
    loss = criterion(mu_pred, Y, sigma_pred**2)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


#Evaluation mode
model.eval()

with torch.no_grad():
    mu_hat, sigma_hat = model(X)

#Numpy conversion
mu_pred = mu_hat.numpy()
sigma_pred = sigma_hat.numpy()


#rain event at 80 mm
rain_event = torch.tensor([[80.0]])
flood_threshold = 110.0

model.eval()
with torch.no_grad():
    mu_val, sigma_val = model(rain_event)

dist = Normal(loc=mu_val, scale=sigma_val)

#Probability of discharge to pass flood_treshold
prob_under_threshold = dist.cdf(torch.tensor([flood_threshold]))
prob_exceedance = 1.0 - prob_under_threshold

#Quantitative resultats
mu = mu_val.item()
sigma = sigma_val.item()
risk_percent = prob_exceedance.item() * 100


print(f"\n--- Flood Scenario: {rain_event.item():.1f} mm precipitation ---")
print(f"Estimated Mean Discharge (μ): {mu:.2f} m³/s")
print(f"Associated Uncertainty (σ): {sigma:.2f} m³/s")
print(f"Critical Levee Threshold: {flood_threshold:.1f} m³/s")
print(f"Exceedance Probability (Overtopping Risk): {risk_percent:.2f}%")



