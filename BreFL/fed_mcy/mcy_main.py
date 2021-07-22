# from torch import std
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from torchvision.transforms import ToTensor
from torchvision import datasets
import random
import numpy as np
from mcy_model import CNNModel, LinearModel
from mcy_server import Server
from mcy_client import Client


# setting
clientNum = 20              # the number of clients
partRate = 0.8              # participation rate
batch_size = 64             # batch size
Epoch = 30                  # number of training epochs

# init the roles of FL
model = CNNModel()          # convolutional neural network
# model = LinearModel()       # linear regression
server = Server(model)      # Parameter Server (PS) architecture
clients = []                # Clients, or "Workers"

transform = transforms.Compose([
    # reference: https://blog.csdn.net/wangkaidehao/article/details/104520022/
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


def initClients():
    print("init clients...")
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,        # transform (aforementioned)
    )
    length = len(training_data)//clientNum      # '//' --> exact division --> can be replaced by 'round(x/y)'
    training_data = random_split(training_data, [length]*clientNum)

    for i in range(clientNum):
        dataloader = DataLoader(training_data[i], batch_size=batch_size)
        client = Client(model, dataloader)
        clients.append(client)
    print("clients inited!!!")


def main():
    initClients()

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for ep in range(Epoch):
        print("round ", ep+1)
        # The 0 step
        perm = list(range(clientNum))
        random.shuffle(perm)
        perm = np.array(perm[:int(clientNum*partRate)])

        # The 1 step
        for client in np.array(clients)[perm]:
            client.getParame(*server.sendParame())
            client.train(1)
            server.getParame(*client.uploadParame())
        
        # The 2 step
        server.aggregate()
        server.test(test_dataloader)


if __name__ == "__main__":
    main()
