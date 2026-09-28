
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch
from Dataset_Dataloader import create_dataloaders
from Neural_network import test_loader_nn, train_loader_nn, val_loader_nn,NeuralNetwork





class DataSetSplit():
    def __init__(self,dataPath):
        self.dataPath=dataPath
        self.data=pd.read_csv(dataPath)

    def split(self):
        x_train,x_temp,y_train,y_temp=train_test_split(self.data.iloc[:,:-1],
                                                       self.data.iloc[:,-1],
                                                       test_size=0.2,
                                                       random_state=42)
        x_val,x_test,y_val,y_test=train_test_split(x_temp,
                                                   y_temp,
                                                   test_size=0.5,
                                                   random_state=42)

        return x_train,x_val,x_test,y_train,y_val,y_test



if __name__ == "__main__":
    dataPath = "data.csv"
    dataset = DataSetSplit(dataPath)
    x_train,x_val,x_test,y_train,y_val,y_test= dataset.split()
    x_train_tensor=torch.tensor(x_train.to_numpy(),dtype=torch.float32)
    x_val_tensor=torch.tensor(x_val.to_numpy(),dtype=torch.float32)
    x_test_tensor=torch.tensor(x_test.to_numpy(),dtype=torch.float32)
    y_train_tensor=torch.tensor(y_train.to_numpy(),dtype=torch.float32)
    y_val_tensor=torch.tensor(y_val.to_numpy(),dtype=torch.float32)
    y_test_tensor=torch.tensor(y_test.to_numpy(),dtype=torch.float32)  
    print("Train and Test Split Completed")

    train_loader, val_loader, test_loader = create_dataloaders(x_train_tensor,y_train_tensor,x_val_tensor,y_val_tensor,x_test_tensor,y_test_tensor)
    model = NeuralNetwork(input_size=x_train_tensor.shape[1],hidden_size=64)
    model.train_loader_nn(train_loader)
    model.val_loader_nn(val_loader)
    model.test_loader_nn(test_loader)

    
