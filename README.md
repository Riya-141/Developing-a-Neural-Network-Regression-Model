# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A company has collected a dataset containing various input features and corresponding numerical output values related to a specific problem (such as sales, price, or demand prediction). The relationship between the input variables and output is complex and cannot be accurately modeled using simple statistical methods.

To solve this problem, the company wants to develop a neural network-based regression model that can learn patterns from the existing data. The model should be trained using historical data so that it can understand how input features influence the output values.

Once trained, the model will be used to predict continuous numerical outputs for new, unseen data points. This will help the company make better decisions based on accurate predictions.

The goal is to minimize prediction error and improve the model’s performance using appropriate training techniques such as backpropagation and optimization algorithms.

## Neural Network Model
<img width="1120" height="767" alt="image" src="https://github.com/user-attachments/assets/f130210a-7976-43a8-86bb-21ea73bdcbe8" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```
dataset1 = pd.read_csv('exp1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
print(X)
print(y)
```
```
dataset1.head()
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
```
```
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

### Name: RIYA P L

### Register Number: 212223240141

```
# Name: RIYA P L
# Register Number: 212223240141
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here

        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}

  def forward(self,X):
    X=self.relu(self.fc1(X))
    X=self.relu(self.fc2(X))
    X=self.fc3(X)
    return X
```
# Initialize the Model, Loss Function, and Optimizer
```
# Initialize the Model, Loss Function, and Optimizer
# Write your code here
lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop (lig. parameters(), lr=0.001)
```
```
# Name: RIYA P L
# Register Number: 212223240141
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range (epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      lig.history['loss'].append(loss.item())
      if epoch%200==0 :
        print(f'Epoch [{epoch}/{epochs}],Loss: {loss.item()}:.6f)')
```
```
train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```
with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
```
```
loss_df = pd.DataFrame(lig.history)
```
```
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```
```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
### Dataset Information
<img width="359" height="451" alt="image" src="https://github.com/user-attachments/assets/cac25a53-e599-40e0-a71c-b49252b15498" />


### OUTPUT
### Training Loss Vs Iteration Plot
<img width="885" height="573" alt="image" src="https://github.com/user-attachments/assets/f548bcb3-de5d-4f6d-a521-3c064876bf08" />


### New Sample Data Prediction
<img width="218" height="300" alt="image" src="https://github.com/user-attachments/assets/94907ed5-31c9-4a0b-9516-672599133bd4" />

<img width="757" height="306" alt="image" src="https://github.com/user-attachments/assets/9a137156-9b51-427e-b6bc-f57a7181653b" />

<img width="621" height="149" alt="image" src="https://github.com/user-attachments/assets/9b234f0f-d22f-4237-bf22-52330957ed33" />

<img width="837" height="124" alt="image" src="https://github.com/user-attachments/assets/d22622fc-61e1-44ea-b4e4-1230ffa18ecb" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
