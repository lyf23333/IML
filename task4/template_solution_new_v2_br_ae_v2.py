import pandas as pd
import numpy as np
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

TITLE = "ONLY PREDICTOR"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = "output"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ENCODER_FILE = "encoder_v2_0.pth"
GAP_FILE = "gap_v2_0.pth"
PREDICTOR_FILE = "predictor_v2_0.pth"
torch.manual_seed(0)


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
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles",
                                                                                                         axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles",
                                                                                                   axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test


class AutoencoderNet(nn.Module):
    def __init__(self, predictor_layer=[], predictor_decoder_layer=[], dropout=0.5):
        super().__init__()
        # define your model layers here

        # predictor
        layers = []
        for i in range(len(predictor_layer) - 2):
            layers += [nn.Linear(predictor_layer[i], predictor_layer[i + 1]), nn.ELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(predictor_layer[-2], predictor_layer[-1])]
        self.predictor = nn.Sequential(*layers)
        # predictor_decoder
        layers = []
        for i in range(len(predictor_decoder_layer) - 2):
            layers += [nn.Linear(predictor_decoder_layer[i], predictor_decoder_layer[i + 1]), nn.ELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(predictor_decoder_layer[-2], predictor_decoder_layer[-1])]
        self.predictor_decoder = nn.Sequential(*layers)

    def forward(self, x):
        # define your forward pass here
        x = self.predictor(x)
        x = self.predictor_decoder(x)
        return x

    def extraction(self, x):
        # defined in the constructor.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(device).to(torch.float32)
        x = self.predictor(x)
        return x


class PredictorNet(nn.Module):
    def __init__(self, predictor_layer=[], predictor_decoder_layer=[], dropout=0.5):
        super().__init__()
        # define your model layers here

        # predictor
        layers = []
        for i in range(len(predictor_layer) - 1):
            layers += [nn.Linear(predictor_layer[i], predictor_layer[i + 1]), nn.ELU(), nn.Dropout(dropout)]
        # layers += [nn.Linear(predictor_layer[-2], predictor_layer[-1])]
        self.predictor = nn.Sequential(*layers)
        # predictor_decoder
        layers = []
        for i in range(len(predictor_decoder_layer) - 2):
            layers += [nn.Linear(predictor_decoder_layer[i], predictor_decoder_layer[i + 1]), nn.ELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(predictor_decoder_layer[-2], predictor_decoder_layer[-1])]
        self.predictor_decoder = nn.Sequential(*layers)

    def forward(self, x):
        # define your forward pass here
        x = self.predictor(x)
        x = self.predictor_decoder(x)
        return x

    def extraction(self, x):
        # defined in the constructor.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(device).to(torch.float32)
        x = self.predictor(x)
        return x


def trainer(x, y, model, batch_size=64, eval_size=100, n_epochs=20, lr=0.0005, weight_decay=0, retrain=True, patience=5):
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    train_dataset = TensorDataset(x_tr, y_tr)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    criterion = nn.MSELoss(reduction="sum")

    best_valid_loss = float('inf')
    counter = 0
    patience = patience

    if retrain:
        model.to(device)
        model.train(True)
        for epoch in range(n_epochs):
            train_loss_epoch = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y, torch.squeeze(y_pred))
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()

            valid_loss_epoch = 0.0
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    loss = criterion(y, torch.squeeze(y_pred))
                    valid_loss_epoch += loss.item()
            scheduler.step(np.sqrt(valid_loss_epoch / len(val_dataset)))
            if (epoch % 2) == 0:
                print(optimizer.param_groups[0]['lr'])
                # scheduler.step()
                print(f"[{epoch}] Training loss {np.sqrt(train_loss_epoch / len(train_loader)):6.3f}, Validation loss {np.sqrt(valid_loss_epoch / len(val_loader)):6.3f}")
            # Early stopping
            if valid_loss_epoch < best_valid_loss:
                best_valid_loss = valid_loss_epoch
                counter = 0  # Reset counter
            else:
                counter += 1  # Increment counter
                if counter >= patience:
                    print('Early stopping')
                    break  # Break out from the loop


def trainer_sklearn(x, y, model, eval_size=100, nfold=5, all_data = False):
    if all_data:
        w = model.fit(x, y)
    else:
        all_score = np.zeros(nfold)
        kf = KFold(n_splits=nfold)
        for j, (train_index, val_index) in enumerate(kf.split(x)):
            w = model.fit(x[train_index], y[train_index])
            # Predict on the validation data
            y_pred = model.predict(x[val_index])

            # Compute and print the R2 score
            score = np.sqrt(np.sum(np.linalg.norm(y[val_index] - y_pred))/y_pred.shape[0])
            all_score[j] = score
            print(f"mean square error {j}th round: {score}")
        print(f"mean square error over all: {np.mean(all_score)}")
    return model

if __name__ == '__main__':
    start = time.time()
    # ============
    # Load data
    # ============
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")

    # ============
    # train autoencoder
    # ============
    predictor_layer = [1000, 512]
    ae_predictor_decoder_layer = predictor_layer[::-1]

    # Train Predictor
    Autoencoder = AutoencoderNet(predictor_layer=predictor_layer, predictor_decoder_layer=ae_predictor_decoder_layer, dropout=0.5)
    # save weights
    print("-----------------------------Training autocoender------------------")
    trainer(x_pretrain, x_pretrain, Autoencoder, batch_size=32, n_epochs=50, lr=0.0002, retrain=True, eval_size=5000, patience=10)

    # ============
    # train Predictor
    # ============

    predictor_layer = [512, 128]
    predictor_decoder_layer = [128, 1]

    Autoencoder.eval()
    with torch.no_grad():
        x_embedding = Autoencoder.extraction(x_pretrain)

    print("-----------------------------Training Predictor------------------")
    Predictor = PredictorNet(predictor_layer=predictor_layer, predictor_decoder_layer=predictor_decoder_layer, dropout=0.5)
    retrain = True
    # save weights
    if retrain:
        trainer(x_embedding, y_pretrain, Predictor, batch_size=32, n_epochs=30, lr=0.0005, retrain=retrain, eval_size=5000, patience=10)
        torch.save(Predictor.state_dict(), os.path.join(OUT_DIR, PREDICTOR_FILE))
        print("save predictor weights. ")
    else:
        Predictor.load_state_dict(torch.load(os.path.join(OUT_DIR, PREDICTOR_FILE)))

    # ============
    # train Gapnet
    # ============

    Predictor.eval()
    Autoencoder.eval()
    with torch.no_grad():
        x_embedding_ae = Autoencoder.extraction(x_train)
        x_embedding = Predictor.extraction(x_embedding_ae)
        x_embedding = x_embedding.clone().detach().cpu().numpy()
    Gap = linear_model.BayesianRidge()  

    # ## Gaussian process 0.226 best
    # kernel_rbf = RBF(length_scale=2.0)
    # kernel_constant = ConstantKernel(constant_value=0.5)
    # kernel_white = WhiteKernel(noise_level=0.2)
    # kernel = kernel_rbf + kernel_constant + kernel_white
    # Gap = GaussianProcessRegressor(kernel=kernel)

    Gap = trainer_sklearn(x_embedding, y_train, Gap, eval_size=10, nfold=10, all_data=False)

    # ============
    # predict
    # ============
    Predictor.eval()
    with torch.no_grad():
        x_embedding_ae = Autoencoder.extraction(x_test.to_numpy())
        x_embedding = Predictor.extraction(x_embedding_ae)
        x_embedding = x_embedding.clone().detach().cpu().numpy()
        y_pred = Gap.predict(x_embedding)

    end = time.time()
    print(f"Time spent: {(end-start)}s")

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")

