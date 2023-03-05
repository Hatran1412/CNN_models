import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 
import torchvision 
import torch.nn as nn
from sklearn.model_selection import train_test_split
import cv2
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class DogCatDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.list_images_path = []
    self.list_labels = []
    self.one_hot_label = {"dogs": 0, "cats": 1}
    for sub_dir in os.listdir(root_dir):
      path_sub_dir = os.path.join(root_dir, sub_dir)
      for image_name in os.listdir(path_sub_dir):
        image_path = os.path.join(path_sub_dir, image_name)
        label = sub_dir
        self.list_images_path.append(image_path)
        self.list_labels.append(label)
    
    self.transform = transform
  
  def __len__(self):
    return len(self.list_images_path)
  
  def __getitem__(self, idx):
    image = cv2.imread(self.list_images_path[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float')
    label = np.array(self.one_hot_label[self.list_labels[idx]]).astype('float')

    sample = (image, label)
    if self.transform:
      sample = self.transform(sample)
    
    return sample # image, label
  
class convertToTensor:
  def __call__(self, sample):
    image, label = sample

    # opencv image: H x W x C
    # torch tensor: C x H x W
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    label = torch.from_numpy(label).long()

    return (image, label)
  
transformed_train_data = DogCatDataset('dog-cat-dataset/data/train', transform=transforms.Compose([convertToTensor()]))
transformed_test_data = DogCatDataset('dog-cat-dataset/data/test', transform=transforms.Compose([convertToTensor()]))

train_data_loader = DataLoader(transformed_train_data, batch_size=32, shuffle=True)
test_data_loader = DataLoader(transformed_test_data, batch_size=32, shuffle=True)


class AlexNet(nn.Module):
  def __init__(self, n_classes):
    super(AlexNet, self).__init__()

    self.feature_extractor = nn.Sequential(
      # convolutional layers 1
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=1),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=1),
      # convolutional layers 2
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=1),
      # convolutional layers 3
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
      nn.ReLU(),
      # convolutional layers 4
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
      nn.ReLU(),
      # convolutional layers 1
      nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )

    self.flatten = nn.Flatten() 

    self.classifier = nn.Sequential(
      nn.Linear(in_features=313632, out_features=256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=128),
      nn.ReLU(),
      nn.Linear(in_features=128, out_features=n_classes), 
    )

  def forward(self, x):
    output = self.feature_extractor(x)
    output = self.flatten(output)
    output = self.classifier(output)

    return output
  

model = AlexNet(2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()

'''
  Function for computing the accuracy of the predictions over the entire data_loader
'''
def get_accuracy(model, data_loader, device):
  correct = 0
  total = 0
  
  with torch.no_grad():
    model.eval()
    for images, labels in data_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return 100*(correct/total)

'''
  Function for plotting training and validation losses
'''
def plot_losses(train_losses, valid_losses):
  # change the style of the plots to seaborn
  plt.style.use('seaborn')

  train_losses = np.array(train_losses)
  valid_losses = np.array(valid_losses)

  fig, ax = plt.subplots(figsize=(8, 4.5))

  ax.plot(train_losses, color="blue", label="Training_loss")
  ax.plot(valid_losses, color="red", label="Validation_loss")
  ax.set(title="Loss over epochs",
          xlabel="Epoch",
          ylabel="Loss")
  ax.legend()
  fig.show()

  # change the plot style to default
  plt.style.use('default')

'''
  function for the training step of the training loop
'''
def train(train_loader, model, criterion, optimizer, device):
  model.train()
  running_loss = 0

  for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    running_loss += loss.item()

    # backward and optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  epoch_loss = running_loss / len(train_loader)

  return model, optimizer, epoch_loss 

'''
  function for the validation step of the training loop
'''
def validate(valid_loader, model, criterion, device):
  model.eval()
  running_loss = 0

  for images, labels in valid_loader:
    images = images.to(device)
    labels = labels.to(device)

    # forward pass and record loss
    outputs = model(images)
    loss = criterion(outputs, labels)
    running_loss = loss.item()
  
  epoch_loss = running_loss / len(valid_loader)

  return model, epoch_loss

'''
  function defining the entire training loop
'''
def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
  # set object for storing metrics
  best_loss = 1e10
  train_losses = []
  valid_losses = []

  # train model
  for epoch in range(0, epochs):
    # training
    model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
    train_losses.append(train_loss)

    # validation
    with torch.no_grad():
      model, valid_loss = validate(valid_loader, model, criterion, device)
      valid_losses.append(valid_loss)

    if epoch % print_every == print_every - 1:
      train_acc = get_accuracy(model, train_loader, device=device)
      valid_acc = get_accuracy(model, valid_loader, device=device)

      print('Epochs: {}, Train_loss: {}, Valid_loss: {}, Train_accuracy: {}, Valid_accuracy: {}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc
            ))

  plot_losses(train_losses, valid_losses)

  return model, optimizer, (train_losses, valid_losses)

model, optimizer, _ = training_loop(model, loss_function, optimizer, train_data_loader, test_data_loader, 50, device)

