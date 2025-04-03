import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, dropout=0.0):
        super().__init__()
        self.arl = arl
        if self.arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0],layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0],layer_sizes[0])
            )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x

class LSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size, hidden_size, output_size], dropout=dropout)

    def forward(self,x,valid_index=None):
        self.lstm.flatten_parameters()
        x,_ = self.lstm(x)
        x = torch.concat([torch.zeros(x.shape[0],1,x.shape[2]).to(x.device),x],dim=1)
        if valid_index is not None:
            x = x[torch.arange(x.size(0)),valid_index]
        else:
            x = x[:, -1, :]
        x = self.fc(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size*2, hidden_size, output_size], dropout=dropout)


    def forward(self,x,valid_index=None):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x1,_ = self.lstm1(x)
        x1 = torch.concat([torch.zeros(x1.shape[0],1,x1.shape[2]).to(x1.device),x1],dim=1)
        if valid_index is not None:
            x1 = x1[torch.arange(x1.size(0)),valid_index]
            x2 = x.clone()
            for i in range(x2.shape[0]):
                end_index = valid_index[i].item()
                x2[i, :end_index] = x[i, :end_index].flip(dims=(0,))
        else:
            x1 = x1[:, -1, :]
            x2 = torch.flip(x, [1])
        x2, _ = self.lstm2(x2)
        x2 = torch.concat([torch.zeros(x2.shape[0],1,x2.shape[2]).to(x2.device),x2],dim=1)
        if valid_index is not None:
            x2 = x2[torch.arange(x2.size(0)),valid_index]
        else:
            x2 = x2[:, -1, :]
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
        return x

class Worker_Net(nn.Module):
    def __init__(self, state_size=7, order_size=4, output_dim=32, bi_direction=False, dropout=0.0):
        super().__init__()
        if bi_direction:
            self.lstm = BiLSTM(order_size,output_dim, dropout=dropout)
        else:
            self.lstm = LSTM(order_size,output_dim, dropout=dropout)

        self.encode = MLP([state_size,output_dim,output_dim], arl=True, dropout=dropout)
        self.mlp = MLP([output_dim*2,output_dim,output_dim], dropout=dropout)

    def forward(self,x_state,x_order,order_num=None):
        x_order = self.lstm(x_order,order_num)
        x_state = self.encode(x_state)
        y = self.mlp(torch.concat([x_state,x_order],dim=-1))
        return y

class Order_Net(nn.Module):
    def __init__(self, state_size=5, output_size=32, dropout=0.0):
        super().__init__()
        self.model = MLP([state_size,output_size,output_size], arl=True, dropout=dropout)

    def forward(self,x):
        y = self.model(x)
        return y

class Attention_Score(nn.Module):
    def __init__(self, input_dims=64, hidden_dims=64, head=1, dropout=0.0, method="mean"):
        super().__init__()
        self.q_linear = nn.ModuleList()
        self.k_linear = nn.ModuleList()
        for i in range(int(head)):
            self.q_linear.append(MLP([input_dims, hidden_dims * 2, hidden_dims * 4], dropout=dropout))
            self.k_linear.append(MLP([input_dims, hidden_dims * 2, hidden_dims * 4], dropout=dropout))
        self.head = head
        self.method = method
        if self.method != "mean":
            self.fuse_layer = MLP([head, head, 1], dropout=dropout)

    def forward(self, q, k):
        attn_matrix = None

        for i in range(self.head):
            query = self.q_linear[i](q)
            key = self.k_linear[i](k)
            key = key ** 2
            norms = torch.norm(key, dim=1, keepdim=True) + 1e-8
            key = key / norms

            attn = torch.mm(query, key.T)
            if self.head == 1:
                return attn

            attn = attn.unsqueeze(-1)
            if attn_matrix is None:
                attn_matrix = attn
            else:
                attn_matrix = torch.concat([attn_matrix, attn], dim=-1)

        if self.method == "mean":
            attn_matrix = torch.mean(attn_matrix, dim=-1)
        else:
            attn_matrix = self.fuse_layer(attn_matrix)
            attn_matrix = attn_matrix.squeeze(-1)
        return attn_matrix

class Q_Net(nn.Module):
    def __init__(self, state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, head=1, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=state_size, order_size=history_order_size, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.order_net = Order_Net(state_size=current_order_size, output_size=hidden_dim, dropout=dropout)
        self.attention = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.attention_price_mu = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.attention_price_sigma = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self,order,x_state,x_order,order_num=None):
        order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()

        order = self.order_net(order)
        worker = self.worker_net(x_state,x_order,order_num)
        q_matrix = self.attention(worker,order)

        price_mu_matrix = self.attention_price_mu(worker,order)
        price_mu_matrix = self.sigmoid(price_mu_matrix) * 2

        price_sigma_matrix = self.attention_price_sigma(worker,order)
        price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.3 + 1e-5

        return q_matrix,price_mu_matrix,price_sigma_matrix

    def feature_freeze(self):
        for layer in self.worker_net.children():
            params = layer.parameters()
            for p in params:
                if p.grad is not None:
                    p.grad[...] = 0

        for layer in self.order_net.children():
            params = layer.parameters()
            for p in params:
                if p.grad is not None:
                    p.grad[...] = 0

class Worker_Q_Net(nn.Module):
    def __init__(self, input_size=14, history_order_size=4, output_dim=2, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=input_size, order_size=history_order_size, output_dim=output_dim, bi_direction=bi_direction, dropout=dropout)

    def forward(self,x,x_order,order_num=None):
        order_num = order_num.int()
        x_order = x_order.float()
        x = x.float()
        y = self.worker_net(x,x_order,order_num) # [reject_prob,accept_prob]
        return y
