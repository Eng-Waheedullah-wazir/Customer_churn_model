import torch
import torch.nn as nn
from Dataset_Dataloader import create_dataloaders

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.fc1=nn.Linear(self.input_size,self.hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(self.hidden_size,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.sigmoid(out)
        return out
  

    def train_loader_nn(self,train_loader): 
        optim=torch.optim.Adam(self.parameters(),lr=0.01)
        loss=nn.BCELoss()
        for epoch in range(10):
            total_loss=0
            for data,labels in train_loader:
                output=self(data)
                labels=labels.view(-1,1)
                loss_value=loss(output,labels)
                optim.zero_grad()
                loss_value.backward()
                optim.step()
                total_loss += loss_value.item()
            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss:{average_loss:.4f}")


      
    def val_loader_nn(self,val_loader):
       loss=nn.BCELoss()
       total_loss=0
       self.eval()
       with torch.no_grad():
           for data,labels in val_loader:
               output=self(data)
               labels=labels.view(-1,1)
               loss_value=loss(output,labels)
               total_loss+=loss_value.item()
           average_loss=total_loss/len(val_loader)
           print(f"Validation Loss:{average_loss:.4f}")
           self.train()
    def test_loader_nn(self,test_loader):
        self.eval()
        loss=nn.BCELoss()
        total_correct=0
        with torch.no_grad():
            for data,labels in test_loader:
                output=self(data)
                labels=labels.view(-1,1)
                loss_value=loss(output,labels)
                total_correct += loss_value.item()
            average_loss=total_correct/len(test_loader)
            print(f"Test Loss:{average_loss:.4f}")
            self.train()
            





