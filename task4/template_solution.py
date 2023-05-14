# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

TITLE = "init"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SuperDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = X
        self.labels = y
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.labels[idx]
        x = self.inputs[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, input_size = 1000, output_size = 1, mlp = [], dropout=0.2, use_bn = False):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        mlp = [input_size] + mlp + [output_size]
        layers =  []
        for i in range(len(mlp)-1):
            if i == len(mlp) - 2:
                self.fe = nn.Sequential(*layers) # feature extractor
                layers += [nn.Linear(mlp[i], mlp[i+1])]
            else:
                if use_bn:
                    layers += [nn.Linear(mlp[i], mlp[i+1]), nn.BatchNorm1d(mlp[i+1]), nn.ELU(), nn.Dropout(dropout)]
                else:
                    layers += [nn.Linear(mlp[i], mlp[i+1]), nn.ELU(), nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.model(x)
        return x
    
    def extraction(self, x):
        """
        Extract features

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, features
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(device).to(torch.float32)
        x = self.fe(x)
        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000, mlp=[512, 256, 128]):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net(input_size=in_features, output_size=1, mlp=mlp)
    model.train()
    model.to(device)
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    train_dataset = SuperDataset(x_tr, y_tr)
    val_dataset = SuperDataset(x_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    n_epochs = 10
    lr = 0.0005

    count = 0
    run_name = f"{TITLE}_{count}"
    while os.path.exists(f"logs/{run_name}"):
        count += 1
        run_name = f"{TITLE}_{count}"
    writer = TensorboardSummaryWriter(log_dir=f"logs/{run_name}", flush_secs=10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    criterion = nn.MSELoss()
    start = time.time()
    # Update the model using optimizer and criterion 
    for epoch in range(n_epochs):
        loss_train, epi_train = run_model(model, train_loader, criterion, optimizer, train=True, device = device)
        loss_val, epi_val= run_model(model, val_loader, criterion, optimizer, train=False, device = device)
        scheduler.step()

        # tensorboard write
        writer.add_scalar("Loss/train", loss_train/epi_train, epoch)
        writer.add_scalar("Loss/val", loss_val/epi_val, epoch)
        end = time.time()
        print("")
        print(f"-------------Epoch {epoch}, spent {end-start}s----------------")
        print(f"Training Loss: {loss_train/epi_train}")
        print(f"Validation Loss: {loss_val/epi_val}")
        print("")


    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        x = model.extraction(x)
        return x

    return make_features

def run_model(model, loader, criterion, optimizer=None, train=True, device=None):
    """
    run model in each epoch
    """
    # Set the model to training mode
    if train:
        model.train()
    else:
        model.eval()
    # Train one epoch
    loss_epoch = 0
    epi = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            # Zero the gradients
            optimizer.zero_grad()
        # Forward pass
        output = model(x)
        # Compute the loss
        loss = criterion(output[:,0], y)
        # Backward pass
        if train:
            loss.backward()
            optimizer.step()

        epi+=1
        loss_epoch+=loss.item()
    return loss_epoch, epi


def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures


def get_regression_model(input_size, mlp=[], dropout = 0.2, use_bn = False):
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = None
    # Define your PyTorch model
    class PyTorchModel(nn.Module):
        def __init__(self, mlp=mlp, input_size=input_size, dropout = dropout, use_bn=use_bn):
            super(PyTorchModel, self).__init__()
            # define your model layers here
            mlp = [input_size] + mlp + [1]
            layers =  []
            for i in range(len(mlp)-1):
                if i == len(mlp) - 2:
                    self.fe = nn.Sequential(*layers) # feature extractor
                    layers += [nn.Linear(mlp[i], mlp[i+1])]
                else:
                    if use_bn:
                        layers += [nn.Linear(mlp[i], mlp[i+1]), nn.BatchNorm1d(mlp[i+1]), nn.ELU(), nn.Dropout(dropout)]
                    else:
                        layers += [nn.Linear(mlp[i], mlp[i+1]), nn.ELU(), nn.Dropout(dropout)]
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            # define your forward pass here
            return self.model(x)

    # Define a wrapper class to make it work with scikit-learn
    class PyTorchModelWrapper(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.model = PyTorchModel()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        def fit(self, X, y=None):
            # Implement fit method which trains the model
            # Convert X and y to PyTorch tensors and move to the correct device, then train
            return self

        def transform(self, X):
            # Implement transform method which applies the model to the data
            # Convert X to PyTorch tensor and move to the correct device, then apply model
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X)
            return predictions
        
        def predict(self, X):
            # The predict method usually just calls the transform method
            return self.transform(X).cpu().numpy()
    model = PyTorchModelWrapper()

    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain, batch_size=64, mlp=[512, 256, 128])
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model(input_size=128, mlp = [64])

    # y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    # Create a pipeline
    pipe = Pipeline([
        ('feature_extraction', PretrainedFeatureClass(feature_extractor = "pretrain")),  # Extract polynomial features
        ('regressor', regression_model)  # Apply Linear Regression
    ])
    
    # Fit the pipeline on the training set
    def custom_score(pipe, x, y_true):
        # Implement your scoring method here
        y_pred = pipe.predict(x)
        return mean_squared_error(y_true.cpu().numpy(), y_pred, squared=False)

    x_train = torch.tensor(x_train).to(device).to(torch.float32)
    y_train = torch.tensor(y_train).to(device).to(torch.float32)
    scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring=custom_score)

    print("Cross-validation scores: ", scores)
    print("Mean cross-validation score: ", scores.mean())

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test.to_numpy())[:,0]

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")