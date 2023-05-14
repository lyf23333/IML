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
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, resnet101, resnet152

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MARGIN = 1
MARGIN_SEMI = 10

def generate_embeddings(model, embedding_name):
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
    # model = resnet50(pretrained=True)
    model = model(pretrained=True)

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

    np.save(f'task3/dataset/embeddings_{embedding_name}.npy', embeddings)


def get_data(file, embedding_file, train=True):
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
    embeddings = np.load(embedding_file)
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
        # if train:
        #     X.append(np.hstack([emb[0].reshape(-1), emb[2].reshape(-1), emb[1].reshape(-1)]))
        #     y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

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
    EMBEDDING_DIM = 128
    
    def __init__(self, inp = 2048, hidden=1024, hidden_1=512, hidden_2=256, hidden_3=128, d=0.5):
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

def train_model(train_loader, val_loader, final_train=False):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 20
    title = "bn"

    count = 0
    run_name = f"resnet_{title}_{count}"
    if os.path.exists(f"task3/logs/{run_name}"):
        count += 1
    run_name = f"resnet_{title}_{count}"
    writer = TensorboardSummaryWriter(log_dir=f"task3/logs/{run_name}", flush_secs=10)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    criterion = nn.TripletMarginLoss(margin=MARGIN)
    criterion_semi = nn.TripletMarginLoss(margin=MARGIN_SEMI)
    start = time.time()
    # Update the model using optimizer and criterion 
    for epoch in range(n_epochs):

        loss_train, epi_train, predictions_train = run_model(model, train_loader, criterion, criterion_semi, optimizer, train=True)
        if not final_train:
            loss_val, epi_val, predictions_val = run_model(model, val_loader, criterion, criterion_semi, optimizer, train=False)
        scheduler.step()

        # tensorboard write
        writer.add_scalar("Loss/train", loss_train/epi_train, epoch)
        if not final_train:
            writer.add_scalar("Loss/val", loss_val/epi_val, epoch)
            writer.add_scalar("Loss/val_accuracy", torch.sum(predictions_val)/predictions_val.shape[0], epoch)
        end = time.time()
        print("")
        print(f"-------------Epoch {epoch}, spent {end-start}s----------------")
        print(f"Training Loss: {loss_train/epi_train}")
        if not final_train:
            print(f"Validation Loss: {loss_val/epi_val}")
            print(f"Validation Accuracy: {torch.sum(predictions_val)/predictions_val.shape[0]}")
            print(f"Train Accuracy: {torch.sum(predictions_train) / predictions_train.shape[0]}")
        print("")

    return model

def select_triplet(x1_embed, x2_embed, x3_embed, y):
    select_indices = []
    for iii,_ in enumerate(x1_embed):
        ab = torch.linalg.norm(x1_embed[iii] - x2_embed[iii])
        ac = torch.linalg.norm(x1_embed[iii] - x3_embed[iii])

        # if ac <= ab + MARGIN: # choose hard triplets and semi-hard triplets
        if (ab <= ac) and (ac < ab + MARGIN):   # choose semi-hard triplets
            select_indices.append(iii)
    select_indices = torch.tensor(select_indices).to(device)
    if select_indices.size(0) > 0:
        return torch.index_select(x1_embed, 0, select_indices), torch.index_select(x2_embed, 0, select_indices), torch.index_select(x3_embed, 0, select_indices), torch.index_select(y, 0, select_indices)
    else:
        return x1_embed, x2_embed, x3_embed, y


def run_model(model, loader, criterion, criterion_semi, optimizer=None, train=True):
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
    # debug: record semi-hard nums
    semi_total = 0
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
            x1_embed_semi, x2_embed_semi, x3_embed_semi, y_semi = select_triplet(x1_embed, x2_embed, x3_embed, y.to(device))
            if (x1_embed_semi.size(0) > 0):
                loss = criterion(x1_embed, x2_embed, x3_embed) #+ criterion_semi(x1_embed_semi, x2_embed_semi, x3_embed_semi)
            else:
                loss = criterion(x1_embed, x2_embed, x3_embed)

        else:
            loss = criterion(x1_embed, x2_embed, x3_embed)

        for iii,_ in enumerate(x1_embed):
            ab = torch.linalg.norm(x1_embed[iii] - x2_embed[iii])
            ac = torch.linalg.norm(x1_embed[iii] - x3_embed[iii])

            if ab > ac:
                predictions.append(0)
            else:
                predictions.append(1)

            # debug: record semi-hard nums
            if (ab <= ac) and (ac < ab + MARGIN):
                semi_total += 1

        error = torch.tensor(predictions).to(y.device) - y == 0
        prediction_results = torch.cat((prediction_results, error))


        # Backward pass
        if train:
            loss.backward()
            optimizer.step()

        epi+=1
        loss_epoch+=loss.item()
    
    # debug: record semi-hard nums
    if train:
        print("semi total in train set: ", semi_total / len(loader.dataset))
    else:
        print("semi total in validation set: ", semi_total / len(loader.dataset))

    return loss_epoch, epi, prediction_results

def test_model(model, loader, save_file):
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
                ac = torch.linalg.norm(predicted_1[iii] - predicted_3[iii])

                if (ab > ac):
                    predictions.append(0)
                else:
                    predictions.append(1)
            # Rounding the predictions to 0 or 1
            # predicted[predicted >= 0.5] = 1
            # predicted[predicted < 0.5] = 0
            # predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt(save_file, predictions, fmt='%i')

def split_dataset(dataset_file:str, val_size = 300):
    triplets = np.loadtxt(dataset_file)
    triplets_mask = triplets.copy()
    triplets_val = []
    triplets_train = []
    for i in range(triplets.shape[0]):
        if np.any(np.isnan(triplets_mask[i])):
            continue
        triplets_val.append(triplets[i])
        triplets_mask[np.where(triplets == triplets[i][0])] = np.NaN
        triplets_mask[np.where(triplets == triplets[i][1])] = np.NaN
        triplets_mask[np.where(triplets == triplets[i][2])] = np.NaN
        if len(triplets_val) > val_size:
            break
    for i in range(triplets.shape[0]):
        if not np.any(np.isnan(triplets_mask[i])):
            triplets_train.append(triplets[i])
    print(len(triplets_train), len(triplets_val))
    np.savetxt("./task3/validation_triplets_split.txt", np.array(triplets_val, dtype=int), fmt='%05d', delimiter=" ")
    np.savetxt("./task3/train_triplets_split.txt", np.array(triplets_train, dtype=int), fmt='%05d', delimiter=" ")


def voting(results_list: list):
    predictions = np.zeros(59544)
    for results in results_list:
        predictions += np.loadtxt(results)
    predictions_final = predictions.copy()
    predictions_final[predictions > 1] = 1
    predictions_final[predictions <= 1] = 0
    print(np.count_nonzero(predictions_final) / predictions.shape[0])
    np.savetxt('task3/results.txt', predictions_final.reshape([59544, 1]), fmt='%i')


    
# Main function. You don't have to change this
if __name__ == '__main__':
    embedding_list = ['resnet50', 'resnet101', 'resnet152']
    model_list = {"resnet50": resnet50, "resnet101": resnet101, "resnet152": resnet152}
    result_list = []
    for embedding in embedding_list:
        TRAIN_TRIPLETS_ALL = 'task3/train_triplets.txt'
        TRAIN_TRIPLETS = 'task3/train_triplets_split.txt'
        VAL_TRIPLETS = 'task3/validation_triplets_split.txt'
        TEST_TRIPLETS = 'task3/test_triplets.txt'
        save_file = 'task3/results_' + embedding + '.txt'
        embedding_file = 'task3/dataset/embeddings_' + embedding + '.npy'
        result_list.append(save_file)
        final_train = True

        # generate embedding for each image in the dataset
        if (os.path.exists(embedding_file) == False):
            generate_embeddings(model=model_list[embedding], embedding_name=embedding)
        print(f"find embedding {embedding}")

        # load the training and testing data
        if final_train:
            X_train, y_train = get_data(TRAIN_TRIPLETS_ALL, embedding_file)
        else:
            X_train, y_train = get_data(TRAIN_TRIPLETS, embedding_file)
        X_val, y_val = get_data(VAL_TRIPLETS, embedding_file)
        X_test, _ = get_data(TEST_TRIPLETS, embedding_file, train=False)
        # Create data loaders for the training and testing data
        train_loader = create_loader_from_np(X_train, y_train, train=True, batch_size=64)
        val_loader = create_loader_from_np(X_val, y_val, train=True, batch_size=64)
        test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)
        # define a model and train it
        model = train_model(train_loader, val_loader, final_train=final_train)

        # test the model on the test data
        test_model(model, test_loader, save_file)
        print("Results saved to " + save_file)

    voting(result_list)



# if __name__ == '__main__':
#     TRAIN_TRIPLETS = 'task3/train_triplets.txt'
#     TEST_TRIPLETS = 'task3/test_triplets.txt'
#
#     split_dataset(TRAIN_TRIPLETS)

