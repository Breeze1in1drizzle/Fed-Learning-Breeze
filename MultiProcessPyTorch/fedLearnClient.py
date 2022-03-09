'''
模拟联邦学习中的 Client
'''
import torch
from torch import nn, optim

from multiprocessing import Process


class Client:

    def __init__(self, model, dataloader, loss="CrossEntropyLoss", optimizer="SGD") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.Epoch = None
        self.loss = None

        if loss == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss()
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epoch=5):
        size = len(self.dataloader.dataset)

        for _ in range(epoch):
            epoch_loss = 0
            for batch, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            epoch_loss /= size

        self.loss = epoch_loss
        print(f"Client loss: {epoch_loss:>7f}")

    def test(self):
        size = len(self.dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f"Client test error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    # get the global model's parameters from parameter server
    def getParame(self, Epoch, parame):
        self.Epoch = Epoch
        self.model.load_state_dict(parame)

    # upload the local model's parameters to parameter server
    def uploadParame(self):
        return self.Epoch, self.model.state_dict(), self.loss

    def start_client(self):
        '''
        这个函数用来start一个client线程
        接受到来自Server的指令后，执行training
        然后把local model上传给Server，等待下一次Server的指令
        '''
        while True:
            pass


if __name__ == "__main__":
    pass
