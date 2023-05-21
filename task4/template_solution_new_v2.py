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


class EncoderNet(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """

    def __init__(self, encoder_layer=[], decoder_layer=[], dropout=0.3):
        """
        The constructor of the model.
        """
        super().__init__()
        # encoder
        layers = []
        for i in range(len(encoder_layer) - 2):
            layers += [nn.Linear(encoder_layer[i], encoder_layer[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(encoder_layer[-2], encoder_layer[-1])]
        self.encoder = nn.Sequential(*layers)

        # decoder
        layers = []
        for i in range(len(decoder_layer) - 2):
            layers += [nn.Linear(decoder_layer[i], decoder_layer[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(decoder_layer[-2], decoder_layer[-1])]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # defined in the constructor.
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def extraction(self, x):
        # defined in the constructor.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(device).to(torch.float32)
        x = self.encoder(x)
        return x

class PredictorNet(nn.Module):
    def __init__(self, predictor_layer=[], predictor_decoder_layer=[], dropout=0.5):
        super().__init__()
        # define your model layers here

        # predictor
        layers = []
        for i in range(len(predictor_layer) - 2):
            layers += [nn.Linear(predictor_layer[i], predictor_layer[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(predictor_layer[-2], predictor_layer[-1])]
        self.predictor = nn.Sequential(*layers)
        # predictor_decoder
        layers = []
        for i in range(len(predictor_decoder_layer) - 2):
            layers += [nn.Linear(predictor_decoder_layer[i], predictor_decoder_layer[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
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

class GapNet(nn.Module):
    def __init__(self, model_layer=[], dropout=0.3):
        super().__init__()
        # define your model layers here
        # predictor
        layers = []
        for i in range(len(model_layer) - 2):
            layers += [nn.Linear(model_layer[i], model_layer[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(model_layer[-2], model_layer[-1])]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # define your forward pass here
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(device).to(torch.float32)
        x = self.model(x)
        return x



def trainer(x, y, model, batch_size=64, eval_size=100, n_epochs=20, lr=0.0005, weight_decay=0, retrain=True):
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
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5)
    criterion = nn.MSELoss(reduction="sum")

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
            if (epoch % 4) == 0:
                print(optimizer.param_groups[0]['lr'])
                # scheduler.step()
                print(f"[{epoch}] Training loss {np.sqrt(train_loss_epoch / len(train_dataset)):6.3f}, Validation loss {np.sqrt(valid_loss_epoch / len(val_dataset)):6.3f}")

if __name__ == '__main__':
    # ============
    # Load data
    # ============
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")

    # ============
    # train
    # ============

    # encoder_layer = [1000, 2048, 1024, 512]
    # decoder_layer = [512, 1024, 2048, 1000]
    predictor_layer = [1000, 64, 16, 4]
    predictor_decoder_layer = [4, 1]
    model_layer = [4, 2, 1]

    # # Train Encoder
    # Encoder = EncoderNet(encoder_layer=encoder_layer, decoder_layer=decoder_layer)
    # retrain = True
    # if retrain:
    #     trainer(x_pretrain, x_pretrain, Encoder, batch_size=128, n_epochs=50, lr=0.003, retrain=retrain, eval_size=1000)
    #     # save weights
    #     torch.save(Encoder.state_dict(), os.path.join(OUT_DIR, ENCODER_FILE))
    #     print("save encoder weights. ")
    # else:
    #     Encoder.load_state_dict(torch.load(os.path.join(OUT_DIR, ENCODER_FILE)))
    # # encoder
    # Encoder.eval()
    # with torch.no_grad():
    #     x_embedding = Encoder.extraction(x_pretrain)
    #     x_embedding = x_embedding.clone().detach().cpu().numpy()

    # Train Predictor
    Predictor = PredictorNet(predictor_layer=predictor_layer, predictor_decoder_layer=predictor_decoder_layer)
    retrain = True
    # save weights
    if retrain:
        trainer(x_pretrain, y_pretrain, Predictor, batch_size=64, n_epochs=50, lr=0.003, retrain=retrain, eval_size=1000)
        torch.save(Predictor.state_dict(), os.path.join(OUT_DIR, PREDICTOR_FILE))
        print("save predictor weights. ")
    else:
        Predictor.load_state_dict(torch.load(os.path.join(OUT_DIR, PREDICTOR_FILE)))

    # Train GapNet
    # Encoder & Predictor
    Predictor.eval()
    with torch.no_grad():
        x_embedding = Predictor.extraction(x_train)
        x_embedding = x_embedding.clone().detach().cpu().numpy()
    Gap = GapNet(model_layer=model_layer)
    trainer(x_embedding, y_train, Gap, batch_size=4, n_epochs=29, lr=0.003, retrain=True, weight_decay=0.005, eval_size=10)
    # save weights
    torch.save(Gap.state_dict(), os.path.join(OUT_DIR, GAP_FILE))
    print("save gap weights. ")

    # ============
    # predict
    # ============
    Predictor.eval()
    Gap.eval()
    with torch.no_grad():
        x_embedding = Predictor.extraction(x_test.to_numpy())
        x_embedding = x_embedding.clone().detach().cpu().numpy()
        y_pred = Gap(x_embedding)
        y_pred = y_pred.clone().detach().cpu().numpy()
        y_pred = y_pred[:, 0]


    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")

