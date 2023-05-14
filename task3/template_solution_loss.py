# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import time
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, resnet101, resnet152

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
                              shuffle=False)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    model = resnet152(pretrained=True)

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

def data_augmentation(file, pre_file):
    triplets = []
    with open(pre_file) as f:
        for line in f:
            triplets.append(line)

    with open(file, "w") as f:
        for line in triplets:
            f.write(line)
            first_num = line[:5]
            second_num = line[6:11]
            new_line = second_num + " " + first_num + line[11:]
            f.write(new_line)

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 1):
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
                        shuffle=shuffle)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    EMBEDDING_DIM = 256
    
    def __init__(self, inp = 2048, hidden=1024, hidden_1=512, hidden_2=256, d=0.3):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.PReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(d),
            nn.Linear(hidden, hidden_1),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_1),
            nn.Dropout(d),
            nn.Linear(hidden_1, hidden_2),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_2),
            nn.Dropout(d),
            nn.Linear(hidden_2, Net.EMBEDDING_DIM)
        )
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
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
    title = "data_augmentation"

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
                        shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                        batch_size=loader.batch_size,
                        shuffle=True)
    count = 0
    run_name = f"resnet152_{title}_{count}"
    if os.path.exists(f"task3/logs/{run_name}"):
        count += 1
    run_name = f"resnet152_{title}_{count}"
    writer = TensorboardSummaryWriter(log_dir=f"task3/logs/{run_name}", flush_secs=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    criterion = nn.TripletMarginLoss(margin=1)
    start = time.time()
    # Update the model using optimizer and criterion 
    for epoch in range(n_epochs):
        loss_train, epi_train, _ = run_model(model, train_loader, criterion, optimizer, train=True)
        if not final_train:
            loss_val, epi_val, predictions = run_model(model, val_loader, criterion, optimizer, train=False)
        scheduler.step()

        # tensorboard write
        writer.add_scalar("Loss/train", loss_train/epi_train, epoch)
        if not final_train:
            writer.add_scalar("Loss/val", loss_val/epi_val, epoch)
            writer.add_scalar("Loss/val_accuracy", torch.sum(predictions)/predictions.shape[0], epoch)
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
    prediction_results = torch.tensor([])
    epi = 0
    for X_h, y in loader:
        predictions = []
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
        if train:
            loss = criterion(x1_embed, x2_embed, x3_embed)
        else:
            loss = criterion(x1_embed, x2_embed, x3_embed)
            for iii,_ in enumerate(x1_embed):
                ab = torch.linalg.norm(x1_embed[iii] - x2_embed[iii])
                bc = torch.linalg.norm(x1_embed[iii] - x3_embed[iii])

                if ab > bc:
                    predictions.append(0)
                else:
                    predictions.append(1)
            error = torch.tensor(predictions).to(y.device) - y == 0
            prediction_results = torch.cat((prediction_results, error))

        # Backward pass
        if train:
            loss.backward()
            optimizer.step()

        epi+=1
        loss_epoch+=loss.item()
    return loss_epoch, epi, prediction_results

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
    TRAIN_TRIPLETS_AUG = 'task3/train_triplets_aug.txt'
    TEST_TRIPLETS = 'task3/test_triplets.txt'
    final_train = False

    # generate embedding for each image in the dataset
    if(os.path.exists('task3/dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    # data_augmentation(TRAIN_TRIPLETS_AUG, TRAIN_TRIPLETS)
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