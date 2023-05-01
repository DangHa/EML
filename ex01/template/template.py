
# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch

# Convinience functions
def plot_model(model=None):
    # Visualize data
    plt.plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test, y_test, 'xr', label='Test data')
    # Visualize model
    if model is not None:
        plt.plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    plt.show()

# Generate data
n_samples = 11
noise_amplitude = 0.15

def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    # ---- Fill in the following:
    result = np.sin(2*np.pi*x)
    return result

torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

# Test plotting
plot_model()
plt.savefig('Initial_data.png')
plt.clf()


# Model fitting

def error_function(model, x_data, y_data):
    y_pred = model(x_data)
    # ---- Fill with the error function from the lecture
    error = torch.sqrt(torch.sum((y_pred - y_data)**2)/len(y_pred))
    return error

model_degree = 11

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"{train_err=}, {test_err=}")

# Result plotting
plot_model(model)
plt.savefig('Initial_fit.png')
plt.clf()

# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size
# Polynomial degree
n_degree = 12
train_arr = []
test_arr = []

for i in range(n_degree):
    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=i)
    train_arr.append(error_function(model, x_train, y_train))
    test_arr.append(error_function(model, x_test, y_test))

plt.plot(range(n_degree), train_arr, 'b-', label="Training loss")
plt.plot(range(n_degree), test_arr, 'r-', label="Testing loss")
plt.title("Polynomial degree against the train and test error")
plt.legend()
plt.savefig('Polynomial_degree.png')
plt.clf()

# Data size
poly_degree = 10
sample_numer = 10
omega = 0.001

while True:
    x_test = torch.linspace(0, 1, sample_numer)
    y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(sample_numer,))
    x_train = torch.linspace(0, 1, sample_numer)
    y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(sample_numer,))

    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=poly_degree)
    train_err = error_function(model, x_train, y_train)
    test_err = error_function(model, x_test, y_test)

    if torch.abs(train_err - test_err) < omega:
        print("\n######################################\n")
        print("The number of samples which has the same training and testing error is: ", sample_numer)
        print(f"{train_err=}, {test_err=}")
        break

    sample_numer += 1