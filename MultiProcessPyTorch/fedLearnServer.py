import torch
from torch import nn
import random
import copy

from multiprocessing import Process
from multiprocessing import Queue


class Server:
    def __init__(self, globalModel, loss="CrossEntropyLoss", parallel=False) -> None:
        self.parallel = parallel        # 是否使用并行simulation模式
        # self.model_queue = Queue(maxsize=-1)  # 多线程并行模拟FL的时候，用于收集Local Models的队列
        self.state = 0      # 指定当前状态    0-->local training, 1-->model aggregation
        # 在model aggregation阶段，client等待（或者说休眠）
        # （这里也可以把模型聚合的耗时算上？——其实不需要，因为时长是固定的，这个不是重点）

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
        print(f"Server test error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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

    def start_server(self):
        '''
        这个函数用来start一个server线程
        首先开启一个循环
        server状态设置为训练
        然后
        循环里面
        '''
        while True:
            pass


if __name__ == "__main__":
    pass