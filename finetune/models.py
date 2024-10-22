import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, dropout=0.0):
        super().__init__()
        self.arl = arl
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
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2 ),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2 , output_size),
        # )
        self.fc = MLP([hidden_size, hidden_size // 2, output_size], dropout=dropout)

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
        self.lstm1 = nn.LSTM(input_size, hidden_size // 2, batch_first=True, bidirectional=False, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size, hidden_size // 2, batch_first=True, bidirectional=False, dropout=dropout)

        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2 ),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2 , output_size),
        # )
        self.fc = MLP([hidden_size, hidden_size // 2, output_size], dropout=dropout)


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
            self.lstm = BiLSTM(order_size,32, dropout=dropout)
        else:
            self.lstm = LSTM(order_size,32, dropout=dropout)

        # self.encode = nn.Sequential(
        #     nn.Linear(state_size,16),
        #     nn.ReLU(),
        #     nn.Linear(16,32)
        # )
        self.encode = MLP([state_size,16,32], arl=True, dropout=dropout)

        self.mlp = MLP([64,64,32,output_dim], dropout=dropout)

    def forward(self,x_state,x_order,order_num=None):
        x_order = self.lstm(x_order,order_num)
        x_state = self.encode(x_state)
        # print(x_state.shape,x_order.shape)
        y = self.mlp(torch.concat([x_state,x_order],dim=-1))
        return y

class Order_Net(nn.Module):
    def __init__(self, state_size=5, output_size=32, dropout=0.0):
        super().__init__()
        self.model = MLP([state_size,16,32,output_size], arl=True, dropout=dropout)

    def forward(self,x):
        y = self.model(x)
        return y

class Attention_Score(nn.Module):
    def __init__(self, input_dims=64,hidden_dims=64,head=2, dropout=0.0):
        super().__init__()
        # self.q_linear=nn.Linear(input_dims,hidden_dims)
        # self.k_linear=nn.Linear(input_dims,hidden_dims)
        self.q_linear = MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout)
        self.k_linear = MLP([input_dims,hidden_dims,hidden_dims], dropout=dropout)
        self.head=head
        self.num=hidden_dims//head
        self.input_dims=input_dims


    def forward(self,q,k):
        query=self.q_linear(q)
        key=self.k_linear(k)

        query=query.view(-1,self.head,self.num).transpose(0, 1)
        key=key.view(-1,self.head,self.num).transpose(0, 1)

        attn_matrix=torch.bmm(query,key.transpose(1, 2))
        attn_matrix=torch.sum(attn_matrix,dim=0)

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
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.attention_time = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)
        self.attention_time_sigma = Attention_Score(input_dims=hidden_dim,hidden_dims=hidden_dim,head=head, dropout=dropout)


    def forward(self,order,x_state,x_order,order_num=None):
        order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()

        order = self.order_net(order)
        worker = self.worker_net(x_state,x_order,order_num)
        q_matrix = self.attention(worker,order)

        price_mu_matrix = self.attention_price_mu(worker,order)
        # price_mu_matrix = price_mu_matrix ** 2
        price_mu_matrix = self.sigmoid(price_mu_matrix) * 2
        # price_mu_matrix = self.sigmoid(price_mu_matrix) + 0.5
        # price_mu_matrix = self.relu(self.tanh(price_mu_matrix) + 0.5)
        # price_mu_matrix = self.tanh(price_mu_matrix)
        # price_mu_matrix = self.relu(price_mu_matrix)


        price_sigma_matrix = self.attention_price_sigma(worker,order)
        # price_sigma_matrix = price_sigma_matrix ** 2 + 1e-8
        # price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.1
        # price_sigma_matrix = self.relu(price_sigma_matrix) + 1e-8
        # price_sigma_matrix = self.softplus(price_sigma_matrix)
        # price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.5
        # price_sigma_matrix = self.softplus(price_sigma_matrix) * 0.5
        # price_sigma_matrix = self.softplus(price_sigma_matrix) + 1e-5
        price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.3 + 1e-5
        # price_sigma_matrix = self.sigmoid(price_sigma_matrix) * 0.5 + 1e-5

        # price_sigma_matrix = self.sigmoid(price_sigma_matrix)
        # price_sigma_matrix = price_sigma_matrix * price_mu_matrix + 1e-5

        time_prediction_mu = self.attention_time(worker,order)
        time_prediction_mu = self.relu(time_prediction_mu)
        time_prediction_sigma = self.attention_time(worker,order)
        time_prediction_sigma = self.softplus(time_prediction_sigma)

        return q_matrix,price_mu_matrix,price_sigma_matrix,[time_prediction_mu,time_prediction_sigma]

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
    def __init__(self, input_size=14, history_order_size=4, output_dim=2, bi_direction=False, dropout=0.0): # (7+1)+(5+1)=14
        super().__init__()
        self.worker_net = Worker_Net(state_size=input_size, order_size=history_order_size, output_dim=output_dim, bi_direction=bi_direction, dropout=dropout)

    # def forward(self,order,price,x_state,reservation_value,x_order,order_num=None):
    #     order_num = order_num.int()
    #     x_state = x_state.float()
    #     x_order = x_order.float()
    #     order = order.float()
    #     price = price.float()
    #     reservation_value = reservation_value.float()
    #     x_state = torch.concat([x_state,reservation_value,order,price],dim=-1)
    #     y = self.worker_net(x_state,x_order,order_num)
    #     return y

    def forward(self,x,x_order,order_num=None):
        # print(x.shape,x_order.shape,order_num.shape)
        order_num = order_num.int()
        x_order = x_order.float()
        x = x.float()
        y = self.worker_net(x,x_order,order_num) # [reject_prob,accept_prob]
        return y


# test
if __name__ == '__main__':
    # model = LSTM(10,1,64)
    # x = torch.randn([4,50,10])
    # index = torch.tensor([5,3,2,1],dtype=torch.int)
    # y = model(x,index)
    # print(y.shape)

    # model = BiLSTM(10, 1, 64)
    # x = torch.randn([4, 50, 10])
    # # index = torch.tensor([5, 3, 2, 1], dtype=torch.int)
    # # y = model(x, index)
    # y = model(x)
    # print(y.shape)

    # model = Worker_Net(5)
    # x_state = torch.randn([2,5])
    # x_order = torch.randn([2,10,4])
    # order_num =  torch.tensor([5,3],dtype=torch.int)
    # y = model(x_state,x_order,order_num)
    # print(y.shape)

    model = Q_Net()
    optm = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    x_state = torch.randn([2,7])
    x_order = torch.randn([2,10,4])
    order_num =  torch.tensor([5,3],dtype=torch.int)
    order_state = torch.randn([4,5])
    q_value, price_mu, price_sigma = model(order_state,x_state,x_order,order_num)
    # print(q_value.shape)
    # print(price_mu.shape)
    # print(price_sigma.shape)
    y = torch.zeros_like(q_value)
    loss = loss_func(q_value,y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(set(model.attention_price_mu.parameters())|set(model.attention_price_sigma.parameters()), 1.0)

    for layer in model.worker_net.children():
        params = layer.parameters()
        for p in params:
            print(p.grad)

    model.feature_freeze()

    for layer in model.worker_net.children():
        params = layer.parameters()
        for p in params:
            print(p.grad)

    optm.step()
