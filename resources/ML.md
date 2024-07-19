

# Machine Learning and AI Libraries Cheatsheet

## TensorFlow

### Description
TensorFlow is an end-to-end open-source platform for machine learning. It provides a comprehensive ecosystem of tools, libraries, and community resources for researchers and developers[1][2].

### Installation
```bash
pip install tensorflow
```

### Basic Usage
```python
import tensorflow as tf

# Create a simple neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)
```

### Key Features
- Flexible ecosystem for ML development
- Supports deployment on various platforms (cloud, on-premise, mobile, edge devices)
- Provides high-level APIs like Keras for easy model building
- Powerful for both research and production

## PyTorch

### Description
PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It's known for its flexibility and dynamic computational graphs[3].

### Installation
```bash
pip install torch torchvision
```

### Basic Usage
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the network
net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Key Features
- Dynamic computational graphs
- Strong support for GPU acceleration
- Pythonic and intuitive API
- Excellent for research and prototyping

## Keras

### Description
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano[2][4].

### Installation
```bash
pip install keras
```

### Basic Usage
```python
from keras.models import Sequential
from keras.layers import Dense

# Create a simple model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=32)
```

### Key Features
- User-friendly, modular, and extensible
- Supports both convolutional networks and recurrent networks
- Runs seamlessly on CPU and GPU
- Easy to use for both beginners and experts

These libraries are powerful tools for developing machine learning and AI applications. TensorFlow and Keras are often used together, with Keras serving as a high-level API for TensorFlow. PyTorch is particularly popular in research settings due to its flexibility and ease of use for prototyping. Choose the library that best fits your project requirements and personal preferences.

Citations:
[1] https://opensource.google/projects/tensorflow
[2] https://www.tensorflow.org
[3] https://makepath.com/open-source-machine-learning-tools/
[4] https://www.anaconda.com/blog/the-best-open-source-tools-for-machine-learning
[5] https://github.com/EthicalML/awesome-production-machine-learning

