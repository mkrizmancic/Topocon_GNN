# Simple demonstration of a neural network with 1 input, 1 hidden layer (2 neurons, ReLU), and 1 output
# Trains on a single input-output pair, prints output, gradients, and weights at each step

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"  # Use browser for Plotly visualizations.

# Set random seed for reproducibility
torch.manual_seed(8)


# Define the network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)  # 1 input -> 2 hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)  # 2 hidden -> 1 output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the network, loss, and optimizer
net = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Single input-output pair
data_in = torch.tensor([[1.0], [2.0], [1.5]])  # shape (3, 1)
data_out = torch.tensor([[2.0], [4.0], [3.0]]) # shape (3, 1)

outputs = []
losses = []
grads_fc1_weight = []
grads_fc1_bias = []
grads_fc2_weight = []
grads_fc2_bias = []
weights_fc1 = []
weights_fc2 = []
biases_fc1 = []
biases_fc2 = []

for epoch in range(20):
    batch_outputs = []
    batch_losses = []

    # Online processing: iterate over each input-output pair
    # for data_in_batch, data_out_batch in zip(data_in, data_out):
    #     data_in_batch = data_in_batch.unsqueeze(0)  # shape (1, 1)
    #     data_out_batch = data_out_batch.unsqueeze(0)  # shape (1, 1)

    #     optimizer.zero_grad()
    #     output = net(data_in_batch)
    #     loss = criterion(output, data_out_batch)
    #     loss.backward()
    #     optimizer.step()

    #     batch_outputs.append(output)
    #     batch_losses.append(loss.item())
    # output = torch.stack(batch_outputs)  # shape (3, 1)
    # loss = np.mean(batch_losses)  # Average loss for the batch

    # Batch processing: stack outputs and compute average loss
    optimizer.zero_grad()
    output = net(data_in)
    loss = criterion(output, data_out)
    loss.backward()
    optimizer.step()

    outputs.append(output.detach().clone().numpy().copy())
    losses.append(loss.item())
    # Handle None gradients by appending zeros
    grads_fc1_weight.append(
        net.fc1.weight.grad.detach().clone().numpy().copy()
        if net.fc1.weight.grad is not None
        else np.zeros_like(net.fc1.weight.detach().numpy())
    )
    grads_fc1_bias.append(
        net.fc1.bias.grad.detach().clone().numpy().copy()
        if net.fc1.bias.grad is not None
        else np.zeros_like(net.fc1.bias.detach().numpy())
    )
    grads_fc2_weight.append(
        net.fc2.weight.grad.detach().clone().numpy().copy()
        if net.fc2.weight.grad is not None
        else np.zeros_like(net.fc2.weight.detach().numpy())
    )
    grads_fc2_bias.append(
        net.fc2.bias.grad.detach().clone().numpy().copy()
        if net.fc2.bias.grad is not None
        else np.zeros_like(net.fc2.bias.detach().numpy())
    )
    weights_fc1.append(net.fc1.weight.detach().clone().numpy().copy())
    weights_fc2.append(net.fc2.weight.detach().clone().numpy().copy())
    biases_fc1.append(net.fc1.bias.detach().clone().numpy().copy())
    biases_fc2.append(net.fc2.bias.detach().clone().numpy().copy())

    # print(f"Epoch {epoch+1}")
    # print(f"  Output: {output}")
    # print(f"  Loss: {loss.item():.4f}")
    # print("  Gradients:")
    # for name, param in net.named_parameters():
    #     if param.grad is not None:
    #         print(f"    {name} grad: {param.grad.data}")
    # print("  Weights after step:")
    # for name, param in net.named_parameters():
    #     print(f"    {name}: {param.data}")
    # print("-"*40)

# Normalize the gradients for better visualization
grads_fc1_weight = nn.functional.normalize(torch.tensor(grads_fc1_weight), dim=1)
grads_fc1_bias = nn.functional.normalize(torch.tensor(grads_fc1_bias), dim=1)
grads_fc2_weight = nn.functional.normalize(torch.tensor(grads_fc2_weight), dim=1)
grads_fc2_bias = nn.functional.normalize(torch.tensor(grads_fc2_bias), dim=1)

# Convert lists to numpy arrays for easier plotting
grads_fc1_weight = np.array(grads_fc1_weight).squeeze()
grads_fc1_bias = np.array(grads_fc1_bias).squeeze()
grads_fc2_weight = np.array(grads_fc2_weight).squeeze()
grads_fc2_bias = np.array(grads_fc2_bias).squeeze()
weights_fc1 = np.array(weights_fc1).squeeze()
weights_fc2 = np.array(weights_fc2).squeeze()
biases_fc1 = np.array(biases_fc1).squeeze()
biases_fc2 = np.array(biases_fc2).squeeze()
outputs = np.array(outputs).squeeze()

# Create subplots: 3 rows, 2 columns
fig = make_subplots(rows=3, cols=2, subplot_titles=(
    'Output and Loss', 'Gradients', 'fc1 Weights', 'fc2 Weights', 'fc1 Biases', 'fc2 Biases'))

# Output and Loss
fig.add_trace(go.Scatter(y=outputs[:,0], mode='lines+markers', name='Output [0]'), row=1, col=1)
fig.add_trace(go.Scatter(y=outputs[:,1], mode='lines+markers', name='Output [1]'), row=1, col=1)
fig.add_trace(go.Scatter(y=outputs[:,2], mode='lines+markers', name='Output [2]'), row=1, col=1)
fig.add_trace(go.Scatter(y=losses, mode='lines+markers', name='Loss'), row=1, col=1)

# Gradients (raw)
fig.add_trace(go.Scatter(y=grads_fc1_weight[:,0], mode='lines+markers', name='fc1.weight[0] grad'), row=1, col=2)
fig.add_trace(go.Scatter(y=grads_fc1_weight[:,1], mode='lines+markers', name='fc1.weight[1] grad'), row=1, col=2)
fig.add_trace(go.Scatter(y=grads_fc1_bias[:,0], mode='lines+markers', name='fc1.bias[0] grad'), row=1, col=2)
fig.add_trace(go.Scatter(y=grads_fc1_bias[:,1], mode='lines+markers', name='fc1.bias[1] grad'), row=1, col=2)
fig.add_trace(go.Scatter(y=grads_fc2_weight[:,0], mode='lines+markers', name='fc2.weight[0] grad'), row=1, col=2)
fig.add_trace(go.Scatter(y=grads_fc2_weight[:,1], mode='lines+markers', name='fc2.weight[1] grad'), row=1, col=2)


# fc1 Weights
fig.add_trace(go.Scatter(y=weights_fc1[:,0], mode='lines+markers', name='fc1.weight[0]'), row=2, col=1)
fig.add_trace(go.Scatter(y=weights_fc1[:,1], mode='lines+markers', name='fc1.weight[1]'), row=2, col=1)

# fc2 Weights
fig.add_trace(go.Scatter(y=weights_fc2[:,0], mode='lines+markers', name='fc2.weight[0]'), row=2, col=2)
fig.add_trace(go.Scatter(y=weights_fc2[:,1], mode='lines+markers', name='fc2.weight[1]'), row=2, col=2)

# fc1 Biases
fig.add_trace(go.Scatter(y=biases_fc1[:,0], mode='lines+markers', name='fc1.bias[0]'), row=3, col=1)
fig.add_trace(go.Scatter(y=biases_fc1[:,1], mode='lines+markers', name='fc1.bias[1]'), row=3, col=1)

# fc2 Biases
fig.add_trace(go.Scatter(y=biases_fc2, mode='lines+markers', name='fc2.bias[0]'), row=3, col=2)

fig.update_layout(height=800, width=1200, title_text="Neural Network Training Progress", showlegend=True)
fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=1, col=2)
fig.update_xaxes(title_text="Epoch", row=2, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=2)
fig.update_yaxes(title_text="Epoch", row=3, col=1)
fig.update_yaxes(title_text="Epoch", row=3, col=2)

fig.show()
