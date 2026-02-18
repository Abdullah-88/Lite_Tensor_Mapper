import torch
from torch import nn, Tensor



class VecDyT(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x


class VecDyGeluSine(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))
        self.beta = nn.Parameter(torch.randn(input_shape))
        self.gamma = nn.Parameter(torch.randn(1))
        self.eta = nn.Parameter(torch.randn(1))
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gamma * self.gelu(self.alpha * x) + self.eta * torch.sin(self.beta * x)

        return x



class TTT(nn.Module):
    def __init__(self, dim: int):

        super(TTT, self).__init__()

        self.mapping = nn.Linear(dim,dim,bias=False)

    def forward(self, in_seq: Tensor) -> Tensor:


        outs = []

        for seq in range(in_seq.size(1)):

            state = in_seq[:,seq,:]
            train_view = state + torch.randn_like(state)
            label_view = state
            loss = nn.functional.mse_loss(self.mapping(train_view), label_view)
            grads = torch.autograd.grad(
                loss, self.mapping.parameters(),create_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.mapping.parameters(), grads):

                    param -= 0.01 * grad

            readout = self.mapping(in_seq[:,seq,:]).detach()
            outs.append(readout)
        out = torch.stack(outs, dim=1)

        return out

class FFUnit(nn.Module):
    def __init__(self,dim):

        super().__init__()

        self.proj =  nn.Linear(dim,dim,bias=False)
        self.modulate = VecDyGeluSine(dim)


    def forward(self, x):

        u, v = x, x

        u = self.modulate(u)
        v = self.proj(v)
        g = u * v

        return g

class LiteTensorMapperBlock(nn.Module):
    def __init__(self, dim, num_patch):

        super().__init__()

        self.norm_1 =  VecDyT(dim)
        self.norm_2 =  VecDyT(dim)
        self.memory = TTT(dim)
        self.feedforward = FFUnit(dim)


    def forward(self, x):


        memorypath, FeedForwardpath = x, x

        memorypath = self.norm_1(memorypath)

        memorypath = self.memory(memorypath)

        FeedForwardpath = self.norm_2(FeedForwardpath)

        FeedForwardpath = self.feedforward(FeedForwardpath)

        x = memorypath + FeedForwardpath

        return x


class LiteTensorMapper(nn.Module):
    def __init__(self, d_model,num_patch, num_layers):
        super().__init__()

        self.model = nn.Sequential(
            *[LiteTensorMapperBlock(d_model,num_patch) for _ in range(num_layers)]
        )

    def forward(self, x):

        return self.model(x)
