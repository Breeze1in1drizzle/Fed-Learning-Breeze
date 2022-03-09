# from torch import std
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from torchvision.transforms import ToTensor
from torchvision import datasets
import random
import numpy as np
from fedLearnDNNModels import CNNModel, LinearModel
from fedLearnServer import Server
from fedLearnClient import Client

# setting
clientNum = 20  # the number of clients     # 如果是多线程模拟，那么这个同时也指的是Clients的线程数
partRate = 0.8  # participation rate
batch_size = 64  # batch size
Epoch = 30  # number of training epochs

# init the roles of FL
model = CNNModel()  # convolutional neural network
# model = LinearModel()     # linear regression
server = Server(model)  # Parameter Server (PS) architecture
clients = []  # Clients, or "Workers"

transform = transforms.Compose([
    # reference: https://blog.csdn.net/wangkaidehao/article/details/104520022/
    transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))
])


def initClients():
    print("init clients...")
    training_data = datasets.MNIST(root="data", train=True,
                        download=True, transform=transform)  # transform (aforementioned)

    length = len(training_data) // clientNum  # '//' --> exact division --> can be replaced by 'round(x/y)'
    training_data = random_split(training_data, [length] * clientNum)       # 数据集拆分-->这里可优化

    for i in range(clientNum):
        dataloader = DataLoader(training_data[i], batch_size=batch_size)
        client = Client(model, dataloader)
        clients.append(client)
    print("clients inited!!!")


def serial_fed_learn_simulation():
    initClients()

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for ep in range(Epoch):
        print("round ", ep + 1)
        # The 0 step
        perm = list(range(clientNum))
        random.shuffle(perm)
        perm = np.array(perm[:int(clientNum * partRate)])

        # The 1 step
        for client in np.array(clients)[perm]:
            client.getParame(*server.sendParame())  # 参数服务器分发Global Model给各Client
            client.train(1)                         # 各Client训练模型
            server.getParame(*client.uploadParame())

        # The 2 step
        server.aggregate()
        server.test(test_dataloader)


############################################################
############################################################
####################并行联邦学习训练模拟#######################
def parallel_fed_learn_simulation():
    # 指定一个线程作为 Parameter Server
    # （Parameter Server肯定得有个全局的队列，每次采集 n 个线程的 Clients 的 Local Models）
    # for循环指定 n 个线程作为 Clients
    # 如何模拟 Parameter Server 对各 Client 的模型聚合？
    initClients()

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    '''
    首先用一个for循环把各client的线程都初始化，后面就不需要再管了，只需要把client的线程写好就行
    先在Client中写线程Process的内容
    然后在Server中写线程Process的内容
    再定义同步方法
    每个communication round中，每个client的clocal epoch可以是若干次，不一定只是一次
    再用一个for循环启动每个client，执行local training
    每个client执行好training后就可以
    '''
    for cn in range(clientNum):
        clients[cn]
    #############################################
    #############################################
    ##############这里需要优化#####################
    for ep in range(Epoch):
        print("round ", ep + 1)
        # The 0 step
        perm = list(range(clientNum))
        random.shuffle(perm)
        perm = np.array(perm[:int(clientNum * partRate)])

        # The 1 step
        for client in np.array(clients)[perm]:
            client.getParame(*server.sendParame())  # 参数服务器分发Global Model给各Client
            client.train(1)  # 各Client训练模型
            server.getParame(*client.uploadParame())

        # The 2 step
        server.aggregate()
        server.test(test_dataloader)
    #############################################
    #############################################
    ##############这里需要优化#####################
####################并行联邦学习训练模拟#######################
############################################################
############################################################


if __name__ == "__main__":
    serial_fed_learn_simulation()
    # parallel_fed_learn_simulation()
