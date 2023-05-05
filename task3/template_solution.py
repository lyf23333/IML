# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import time
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from torchvision.models import resnet50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    train_transforms = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])])
    train_dataset = datasets.ImageFolder(root="task3/dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=16)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    model = resnet50(pretrained=True)

    embeddings = []
    embedding_size = 2048
    # Dummy variable, replace with the actual embedding size once you 
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    model.fc = nn.Sequential()
    model.to(device)
    with torch.no_grad():
        model.eval()
        for i, (images, _)  in enumerate(train_loader):
            images = images.to(device)
            features = model(images)
            embeddings[i * train_loader.batch_size: (i + 1) * train_loader.batch_size] = features.cpu().numpy()

    np.save('task3/dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="task3/dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('task3/dataset/embeddings.npy')
    # TODO: Normalize the embeddings across the dataset
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0].reshape(-1), emb[1].reshape(-1), emb[2].reshape(-1)]))
        y.append(1)
        # Generating negative samples (data augmentation)
    #     if train:
    #         X.append(np.hstack([emb[0].reshape(-1), emb[2].reshape(-1), emb[1].reshape(-1)]))
    #         y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 500)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = torch.sigmoid(x)
        return x

def train_model(loader, final_train=False):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10

    if final_train:
        train_loader = loader
    else:
        dataset = loader.dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        # split all_loader into train_loader and test_loader
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(dataset=train_dataset,
                        batch_size=loader.batch_size,
                        shuffle=True,
                        pin_memory=True, num_workers=loader.num_workers)
        val_loader = DataLoader(dataset=val_dataset,
                        batch_size=loader.batch_size,
                        shuffle=True,
                        pin_memory=True, num_workers=loader.num_workers)
    
    writer = TensorboardSummaryWriter(log_dir="task3", flush_secs=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin=1)
    start = time.time()
    # Update the model using optimizer and criterion 
    for epoch in range(n_epochs):
        loss_train, epi_train = run_model(model, train_loader, criterion, optimizer, train=True)
        if not final_train:
            loss_val, epi_val = run_model(model, val_loader, criterion, optimizer, train=False)

        # tensorboard write
        writer.add_scalar("Loss/train", loss_train/epi_train, epoch)
        if not final_train:
            writer.add_scalar("Loss/val", loss_val/epi_val, epoch)
        end = time.time()
        print("")
        print(f"-------------Epoch {epoch}, spent {end-start}s----------------")
        print(f"Training Loss: {loss_train/epi_train}")
        if not final_train:
            print(f"Validation Loss: {loss_val/epi_val}")
        print("")

    return model

def run_model(model, loader, criterion, optimizer=None, train=True):
    """
    run model in each epoch
    """
    # Set the model to training mode
    if train:
        model.train()
    # Train one epoch
    loss_epoch = 0
    epi = 0
    for X_h, y in loader:
        x1 = X_h[:,:2048]
        x2 = X_h[:,2048:4096]
        x3 = X_h[:,4096:]

        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        x1_embed = model(x1)
        x2_embed = model(x2)
        x3_embed = model(x3)
        # Compute the loss
        loss = criterion(x1_embed, x2_embed, x3_embed)

        # Backward pass
        if train:
            loss.backward()
            optimizer.step()

        epi+=1
        loss_epoch+=loss.item()
    return loss_epoch, epi

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            
            x_b1 = x_batch[:,:2048]
            x_b2 = x_batch[:,2048:4096]
            x_b3 = x_batch[:,4096:]
            x_b1, x_b2, x_b3 = x_b1.to(device), x_b2.to(device), x_b3.to(device)      

            predicted_1 = model(x_b1)
            predicted_2 = model(x_b2)
            predicted_3 = model(x_b3)
            
            for iii,_ in enumerate(predicted_1):
              ab = torch.linalg.norm(predicted_1[iii] - predicted_2[iii])
              bc = torch.linalg.norm(predicted_1[iii] - predicted_3[iii])

              if ab > bc:
                 predictions.append(0)
              else:
                 predictions.append(1)
            # Rounding the predictions to 0 or 1
            # predicted[predicted >= 0.5] = 1
            # predicted[predicted < 0.5] = 0
            # predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("task3/results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'task3/train_triplets.txt'
    TEST_TRIPLETS = 'task3/test_triplets.txt'
    final_train = False

    # generate embedding for each image in the dataset
    if(os.path.exists('task3/dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader, final_train=final_train)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")