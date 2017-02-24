import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import pytorchvisu

import numpy as np

# Get data iterators
# We need transforms.ToTensor to convert PIL image into a Torch Tensor

transform = transforms.Compose([transforms.ToTensor()])

mnist_tr = torch.utils.data.DataLoader(
        datasets.MNIST('../datasets', train=True, download=True, transform=transform), 
        batch_size=100, shuffle=True)
mnist_te = torch.utils.data.DataLoader(
        datasets.MNIST('../datasets', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=True)

# Model definition
def build_model(input_dim, output_dim):
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=True))
    model.add_module("softmax", torch.nn.Softmax())
    return model

# Build MNIST model
model = build_model(784, 10) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.005) 
loss_fn = torch.nn.CrossEntropyLoss()

# Extract the weights and biases
params = model.state_dict()
weights = params['linear.weight']
biases = params['linear.bias']

datavis = pytorchvisu.MnistDataVis()

# Visualisation
def training_step(i, update_test_data, update_train_data):
    _, (x, y) = next(enumerate(mnist_tr)) # Get a shuffled batch
    
    y_pred = model(Variable(x.view(-1,28*28)))

    loss = loss_fn(y_pred, Variable(y))
    model.zero_grad()

    loss.backward()
    optimizer.step()

    if update_train_data:
        w = weights.numpy().reshape(-1)
        b = biases.numpy().reshape(-1)
        w.sort()
        b.sort()
        datavis.append_test_curves_data(i, 0.3, loss.data[0])
        datavis.append_data_histograms(i, w, b)
        x_np = x.numpy()
        y_pred_np = np.argmax(y_pred.data.numpy(), axis=1)
        y_np = y.numpy()
        im = pytorchvisu.numpy_format_mnist_images(x_np, y_pred_np, y_np)
        print(im, x_np, y_pred_np, y_np)
        datavis.update_image1(im)
        print(str(i) + ": training loss: " + str(loss.data[0]))

    if update_test_data:
        _, (xt, yt) = next(enumerate(mnist_te))
        w = weights.numpy().reshape(-1)
        b = biases.numpy().reshape(-1)
        w.sort()
        b.sort()
        yt_pred = model(Variable(xt.view(-1,28*28)))
        loss_t = loss_fn(yt_pred, Variable(yt))
        datavis.append_test_curves_data(i, 0.2, loss_t.data[0])
        xt_np = xt.numpy()
        yt_np = yt.numpy()
        yt_pred_np = np.argmax(yt_pred.data.numpy(), axis=1) 
        im = pytorchvisu.numpy_format_mnist_images(xt_np, yt_pred_np, yt_np)
        datavis.update_image2(im)
        print(str(i) + ": test loss: " + str(loss_t.data[0]))

datavis.animate(training_step, iterations=2000, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=False)
