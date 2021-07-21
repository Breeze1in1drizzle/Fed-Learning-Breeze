import torch
from torch import nn
import random
import copy


class Server:
    def __init__(self, globalModel, loss="CrossEntropyLoss") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.Epoch = random.randint(0, 1e8)
        self.local_state_dict = []
        self.train_loss = 0
        self.model = globalModel.to(self.device)
        self.global_state_dict = copy.deepcopy(globalModel.state_dict())

        if loss == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss()

    def test(self, dataloader):
        size = len(dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= size
        correct /= size
        print(f"Server test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def aggregate(self):
        clientNum = len(self.local_state_dict)
        self.train_loss /= clientNum
        print(f"Clients total error: Avg loss: {self.train_loss:>8f}")

        for layer_name in self.global_state_dict.keys():
            self.global_state_dict[layer_name] = torch.zeros_like(self.global_state_dict[layer_name])
            
            for localParame in self.local_state_dict:
                self.global_state_dict[layer_name].add_(localParame[layer_name])
            
            self.global_state_dict[layer_name].div_(clientNum)

        self.local_state_dict.clear()
        self.train_loss = 0
        self.model.load_state_dict(self.global_state_dict)
        self.Epoch = random.randint(0, 1e8) 

    def sendParame(self):
        return self.Epoch, self.global_state_dict

    def getParame(self, Epoch, localParame, loss):
        if Epoch == self.Epoch:
            self.local_state_dict.append(localParame)
            self.train_loss += loss


if __name__ == "__main__":
    pass