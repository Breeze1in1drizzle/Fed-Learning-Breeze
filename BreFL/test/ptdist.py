# filename 'ptdist.py'
import torch
import torch.distributed as dist


def main(rank, world):
    if rank == 0:
        x = torch.tensor([1., -1.])     # Tensor of interest
        dist.send(x, dst=1)
        print('Rank-0 has sent the following tensor to Rank-1')
        print(x)
    else:
        z = torch.tensor([0., 0.])  # A holder for recieving the tensor
        dist.recv(z, src=0)
        print('Rank-1 has recieved the following tensor from Rank-0')
        print(z)


if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(), dist.get_world_size())
