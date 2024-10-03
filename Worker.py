import math
import numpy as np
import torch
import pandas as pd
from models import Q_Net,Worker_Q_Net
from joblib import Parallel, delayed
import torch.nn as nn
import tqdm
import warnings
# ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
INF = 1e8

# imitate the accept rate
def accept_rate(price=1.0,reservation_value=1.0):
    ratio = price / reservation_value
    return 1/(1+math.exp(-50*(ratio-0.95)))

def plot_accept_rate():
    import matplotlib.pyplot as plt

    def f(x):
        return 1 / (1 + np.exp(-50 * (x - 0.95)))

    x_values = np.linspace(0, 2, 400)
    y_values = f(x_values)

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=r'$f(x) = \frac{1}{1 + e^{-50(x - 0.95)}}$', color='blue')
    plt.title('Function Plot of $f(x)$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.legend()
    plt.grid()
    plt.show()


# lat_min, lat_max = 22.24370366972477, 22.505171559633027
# lon_min, lon_max = 113.93901100917432, 114.26928623853212
# lat_range = lat_max - lat_min
# lon_range = lon_max - lon_min
# wait_max_time = 5
# transportation_max_time = 40
# max_seat = 3
# to make all input around 0-1
def norm(order, x_state, x_order, lat_min = 22.24370366972477, lat_max = 22.505171559633027, lon_min = 113.93901100917432, lon_max = 114.26928623853212, wait_max_time = 5, transportation_max_time = 40, max_seat = 3):

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(order, torch.Tensor):
        order, x_state, x_order = order.clone(), x_state.clone(), x_order.clone()
    else:
        order, x_state, x_order = order.copy(), x_state.copy(), x_order.copy()

    # 1. lat & lon
    order[:,0] = (order[:,0] - lat_min) / lat_range
    order[:,2] = (order[:,2] - lat_min) / lat_range
    order[:,1] = (order[:,1] - lon_min) / lon_range
    order[:,3] = (order[:,3] - lon_min) / lon_range

    x_state[:,0] = (x_state[:,0] - lat_min) / lat_range
    x_state[:,1] = (x_state[:,1] - lon_min) / lon_range

    x_order[:,:,0] = (x_order[:,:,0] - lat_min) / lat_range * (x_order[:,:,0] != 0)
    x_order[:,:,1] = (x_order[:,:,1] - lon_range) / lon_range * (x_order[:,:,0] != 0)

    # 2. time
    order[:,4] = order[:,4] / wait_max_time # max wait time: 5 min
    x_order[:,:,2:] = x_order[:,:,2:] / transportation_max_time # max transportation time: 40min as threshold

    # 3. seat
    x_state[:,2] = x_state[:,2] / max_seat # max seat: 3
    x_state[:,4] = x_state[:,4] / max_seat # max seat: 3

    return order, x_state, x_order

# FIFO Buffer
class Buffer():
    def __init__(self,capacity = 1e5):
        super().__init__()
        self.reset(capacity)

    def reset(self, capacity = None):
        if capacity is not None:
            self.capacity = capacity

        self.num = 0

        # state
        self.worker_state = []
        self.order_state = []
        self.order_num = []
        self.new_order_state = []

        # action
        self.price = [] # also worker extra state
        self.price_log_prob = []

        # △t
        self.delta_t = []

        # next_state
        self.worker_state_next = []
        self.order_state_next = []
        self.order_num_next = []
        self.new_order_state_next = []

        # reward
        self.reward = []

        # worker
        self.reservation_value = [] # worker extra state
        self.worker_action = []
        self.worker_reward = []
        self.price_next = [] # worker extra state (next)

    '''
    input: record = [state, worker_current, action, reward, delta_t, next_state, worker_next]
    
    state = [[observe,current_order,current_order_num,new_orders_state, current_time],speed,capacity,positive_history,negative_history] (platform & worker common state)
    worker_current = [[worker_action, worker_reward, price], reservation_value] (worker extra state & action)
    action = [price, price_log_prob]
    reward
    delta_t
    next_state (same structure as "state") 
    worker_next (same structure as "worker_current") 
    '''
    def append(self,record):
        state, worker_current, action, reward, delta_t, next_state, worker_next = record

        speed, capacity, positive, negative = state[1],state[2],state[3],state[4]
        state = state[0]

        speed_next, capacity_next, positive_next, negative_next = next_state[1],next_state[2],next_state[3],next_state[4]
        next_state = next_state[0]

        if self.num == self.capacity:
            self.worker_state = self.worker_state[1:]
            self.order_state = self.order_state[1:]
            self.order_num = self.order_num[1:]
            self.new_order_state = self.new_order_state[1:]
            self.price = self.price[1:]
            self.price_log_prob = self.price_log_prob[1:]
            self.delta_t = self.delta_t[1:]
            self.worker_state_next = self.worker_state_next[1:]
            self.order_state_next = self.order_state_next[1:]
            self.order_num_next = self.order_num_next[1:]
            self.new_order_state_next = self.new_order_state_next[1:]
            self.reward = self.reward[1:]
            self.reservation_value = self.reservation_value[1:]
            self.worker_action = self.worker_action[1:]
            self.worker_reward = self.worker_reward[1:]
            self.price_next = self.price_next[1:]
        else:
            self.num+=1

        worker_state_temp = state[0][:3].tolist()
        worker_state_temp.extend([speed, capacity, positive, negative])
        self.worker_state.append(worker_state_temp)
        self.order_state.append(state[1].tolist())
        self.order_num.append(state[2])
        self.new_order_state.append(state[3].tolist())
        self.price.append(action[0])
        self.price_log_prob.append(action[1])
        self.delta_t.append(delta_t)
        worker_state_next_temp = next_state[0][:3].tolist()
        worker_state_next_temp.extend([speed_next, capacity_next, positive_next, negative_next])
        self.worker_state_next.append(worker_state_next_temp)
        self.order_state_next.append(next_state[1].tolist())
        self.order_num_next.append(next_state[2])
        self.new_order_state_next.append(next_state[3].tolist())
        self.reward.append(reward)
        self.reservation_value.append(worker_current[1])
        self.worker_action.append(worker_current[0][0])
        self.worker_reward.append(worker_current[0][1])
        self.price_next.append(worker_next[0][1])

    '''
    random sample <size> samples
    return:
    
    line1: platform & worker common state
    line2: action (price is also a part of satet of worker), reward, △t
    line3: platform & worker common state_next
    line4: worker extra state/action/reward/state_next
    '''
    def sampling(self,size,device):
        indices = np.random.randint(0, self.num, size=size)

        worker_state = torch.tensor([self.worker_state[i] for i in indices]).to(device)
        order_state = torch.tensor([self.order_state[i] for i in indices]).to(device)
        order_num = torch.tensor([self.order_num[i] for i in indices]).to(device)
        new_order_state = torch.tensor([self.new_order_state[i] for i in indices]).to(device)
        price = torch.tensor([self.price[i] for i in indices]).to(device)
        price_log_prob = torch.tensor([self.price_log_prob[i] for i in indices]).to(device)
        reward = torch.tensor([self.reward[i] for i in indices]).to(device)
        delta_t = torch.tensor([self.delta_t[i] for i in indices]).to(device)
        worker_state_next = torch.tensor([self.worker_state_next[i] for i in indices]).to(device)
        order_state_next = torch.tensor([self.order_state_next[i] for i in indices]).to(device)
        order_num_next = torch.tensor([self.order_num_next[i] for i in indices]).to(device)
        new_order_state_next = torch.tensor([self.new_order_state_next[i] for i in indices]).to(device)

        reservation_value = torch.tensor([self.reservation_value[i] for i in indices]).to(device)
        worker_action = torch.tensor([self.worker_action[i] for i in indices]).to(device)
        worker_reward = torch.tensor([self.worker_reward[i] for i in indices]).to(device)
        price_next = torch.tensor([self.price_next[i] for i in indices]).to(device)

        return worker_state,order_state,order_num,new_order_state,\
            price, price_log_prob, reward, delta_t, \
            worker_state_next, order_state_next, order_num_next, new_order_state_next, \
            reservation_value, price_next, worker_action, worker_reward



'''
num: worker number
history_num: the number of history positive and negative unit-price for each worker
reservation_value/speed: 1.0 as baseline
capacity: the maximum order number of each worker
'''
class Worker():
    def __init__(self, buffer, lr=0.0001, gamma=0.99, eps_clip=0.2, max_step=60, history_num=3, num=1000, reservation_value=None, speed=None, capacity=None, group=None, device=None, zone_table_path = "../data/zone_table.csv", model_path = None,  worker_model_path = None, njobs=24):
        super().__init__()

        self.buffer = buffer
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.history_num = history_num

        self.max_step = max_step

        self.zone_lookup = pd.read_csv(zone_table_path)
        self.coordinate_lookup = np.array(self.zone_lookup[['lat','lon']])

        self.Q_training = Q_Net(state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, head=2, bi_direction=False, dropout=0.3).to(device)
        self.Q_target = Q_Net(state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, head=2, bi_direction=False, dropout=0.3).to(device)
        # if model_path is not None:
        #     self.Q_target.load_state_dict(torch.load(model_path))
        #     self.Q_training.load_state_dict(torch.load(model_path))
        # if model_path is not None:
        #     self.Q_target.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        #     self.Q_training.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

        self.Worker_Q_training = Worker_Q_Net(input_size=14, history_order_size=4, output_dim=2, bi_direction=False, dropout=0.3).to(device)
        self.Worker_Q_target = Worker_Q_Net(input_size=14, history_order_size=4, output_dim=2, bi_direction=False, dropout=0.3).to(device)
        # if worker_model_path is not None:
        #     self.Worker_Q_training.load_state_dict(torch.load(worker_model_path))
        #     self.Worker_Q_target.load_state_dict(torch.load(worker_model_path))
        # if worker_model_path is not None:
        #     self.Worker_Q_training.load_state_dict(torch.load(worker_model_path,map_location=torch.device('cpu')))
        #     self.Worker_Q_target.load_state_dict(torch.load(worker_model_path,map_location=torch.device('cpu')))

        self.load(model_path,worker_model_path,self.device)
        for param in self.Q_target.parameters():
            param.requires_grad = False
        self.Q_target.eval()
        for param in self.Worker_Q_target.parameters():
            param.requires_grad = False
        self.Worker_Q_target.eval()
        self.update_Qtarget(tau=0.0)

        print('Platform total parameters:', sum(p.numel() for p in self.Q_training.parameters() if p.requires_grad))
        print('Worker total parameters:', sum(p.numel() for p in self.Worker_Q_training.parameters() if p.requires_grad))

        self.optim = torch.optim.Adam(self.Q_training.parameters(), lr=lr)
        self.optim_worker = torch.optim.Adam(self.Worker_Q_training.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        # self.reset(max_step,num,reservation_value, speed, capacity, group)

        self.njobs = njobs

    def reset(self, max_step=60, num=1000, reservation_value=None, speed=None, capacity=None, group=None, train=True):
        if train:
            self.Worker_Q_training.train()
            self.Q_training.train()
        else:
            self.Worker_Q_training.eval()
            self.Q_training.eval()

        self.max_step = max_step
        self.num = num

        if reservation_value is None:
            self.reservation_value = np.array([1.0] * self.num)
        else:
            self.reservation_value = reservation_value

        if speed is None:
            self.speed = np.array([1.0] * self.num)
        else:
            self.speed = speed

        # self.real_reservation_value = self.reservation_value * self.speed
        self.worker_reward = np.array([0.0] * self.num)

        if capacity is None:
            self.capacity = np.array([3.0] * self.num)
        else:
            self.capacity = capacity
        self.max_capacity = np.max(self.capacity)

        if group is None:
            self.group = np.array([0] * self.num)
        else:
            self.group = group


        # self.positive_history = np.zeros([self.num,self.history_num])
        # self.negative_history = np.zeros([self.num, self.history_num])
        # for i in range(self.num):
        #     index_pos=0
        #     index_neg=0
        #     while index_neg<self.history_num or index_pos<self.history_num:
        #         record = np.random.randn((5*self.history_num))
        #         record = np.abs(record*0.05 + self.reservation_value[i])
        #         rand = np.random.rand((5*self.history_num))
        #         for j in range(5*self.history_num):
        #             acc_rate = accept_rate(record[j],self.reservation_value[i])
        #             if rand[j]<=acc_rate and index_pos<self.history_num:
        #                 self.positive_history[i,index_pos] = record[j]
        #                 index_pos+=1
        #             elif rand[j]>acc_rate and index_neg<self.history_num:
        #                 self.negative_history[i,index_neg] = record[j]
        #                 index_neg+=1
        #             if index_pos>=self.history_num and index_pos>=self.history_num:
        #                 break
        # '''
        # use single EMA to replace history record (reduce state space size)
        # '''
        # self.positive_history = np.mean(self.positive_history,axis=-1)
        # self.negative_history = np.mean(self.negative_history,axis=-1)
        self.positive_history = self.reservation_value + np.abs(np.random.randn(self.num)) * 0.005
        self.negative_history = self.reservation_value - np.abs(np.random.randn(self.num)) * 0.005

        '''
        observation space
        0,1: current lat,lon (required to be normalized before inputting to the network, following lat and lon remain same)
        2: remaining order place
        3: state -- 0-available 1-picking 2-full
        4: remaining picking time
        current state space only includes the 0,1,2 items
        '''
        self.observe_space = np.zeros([self.num,5])
        self.observe_space[:,2] = self.capacity

        # allocate a initial location randomly from valid zone
        random_integers = np.random.randint(0, len(self.zone_lookup), size=(self.num))
        self.observe_space[:, :2] = self.coordinate_lookup[random_integers]

        '''
        current orders
        0,1: drop-off lat,lon
        2: remaining transportation time (approximated)
        3: total transportation time (approximated)
        '''
        self.current_orders = np.zeros([self.num,int(self.max_capacity),4])
        self.current_order_num = np.zeros([self.num])

        # some records for simulation
        self.travel_route = [[] for _ in range(self.num)]
        self.travel_time = [[] for _ in range(self.num)]
        self.experience = [[] for _ in range(self.num)] # When each item gets full, it will be added to buffer.
        self.Pass_Travel_Time = []

        self.price = [[] for _ in range(self.num)]
        self.salary = [0.0]*self.num
        self.work_load = [0.0]*self.num
        self.worker_assign_order = [0.0]*self.num
        self.worker_reject_order = [0.0]*self.num


    def observe(self, order, current_time, exploration_rate=0):
        # self.Q_training.eval()
        torch.set_grad_enabled(False)
        # 1. contstruct the worker state
        # print(self.observe_space.shape, self.speed.shape, self.capacity.shape, self.positive_history.shape, self.negative_history.shape)
        worker_state = np.concatenate([self.observe_space[:,:3], np.expand_dims(self.speed, axis=-1), np.expand_dims(self.capacity, axis=-1), np.expand_dims(self.positive_history, axis=-1), np.expand_dims(self.negative_history, axis=-1)],axis=-1)
        # 2. construct the order state
        order_state = np.array(order[['plat','plon','dlat','dlon','minute']])
        order_state[:,-1] = current_time - order_state[:,-1] # waiting time
        # 3. get Q value
        x1, x2, x3 = norm(torch.from_numpy(order_state).to(self.device),torch.from_numpy(worker_state).to(self.device),torch.from_numpy(self.current_orders).to(self.device))
        q_value, price_mu, price_sigma = self.Q_training(x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device))
        exploration_matrix = torch.rand_like(q_value)
        q_value[exploration_matrix<exploration_rate] = INF
        # 4. delete the Q value of not available workers
        q_value[self.observe_space[:,3]!=0] = -INF
        return q_value.cpu().detach().numpy(), price_mu.cpu().detach().numpy(), price_sigma.cpu().detach().numpy(), order_state, worker_state

    def update(self, feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, worker_feed_back_table, current_time, final_step=False):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.positive_history[i], self.negative_history[i], self.speed[i], self.capacity[i], self.travel_route[i], self.travel_time[i], self.experience[i], feedback_table[i], new_route_table[i], new_route_time_table[i], new_remaining_time_table[i], new_total_travel_time_table[i], worker_feed_back_table[i], self.reservation_value[i])
            for i in range(self.num))

        for i in range(len(results)):
            # take some record
            if feedback_table[i] is not None: # assign new order
                self.worker_assign_order[i] += 1
                if feedback_table[i][-1] == -1: # reject order
                    self.worker_reject_order[i] += 1
                else:
                    price = feedback_table[i][1][0]
                    work_load = feedback_table[i][1][2]
                    salary = feedback_table[i][1][3]
                    self.work_load[i] += work_load
                    self.salary[i] += salary
                    self.price[i].append(price)

            # update state
            self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.positive_history[i], self.negative_history[i], self.travel_route[i], self.travel_time[i], self.experience[i] \
                = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][6], results[i][7]
            if results[i][8] is not None:
                self.buffer.append(results[i][8])
            if results[i][9] is not None:
                self.Pass_Travel_Time.extend(results[i][9].tolist())
            self.worker_reward[i] += self.gamma ** current_time * results[i][10]

        # # take the ending into consideration (to do, not sure about the effectiveness)
        # if final_step:
        #     for i in range(self.num):
        #         if len(self.experience[i])>0:
        #             self.experience[i].append(self.max_step-self.experience[i][0][-1])
        #             self.experience[i].append(None)
        #             if len(self.experience[i]) == 5:
        #                 self.buffer.append(self.experience[i])
        #             else:
        #                 print("There is a bug (final experience)!!")
        #     finished_order_time = self.current_orders[i, :, 3]
        #     finished_order_time = finished_order_time[finished_order_time!=0]
        #     self.Pass_Travel_Time.extend(finished_order_time.tolist())


    def save(self, path1, path2):
        torch.save(self.Q_training.state_dict(), path1)
        torch.save(self.Worker_Q_training.state_dict(), path2)

    def load(self, path1 = None, path2 = None, device = torch.device("cpu")):
        if device == torch.device("cpu"):
            if path1 is not None:
                self.Q_target.load_state_dict(torch.load(path1,map_location=torch.device('cpu')))
                self.Q_training.load_state_dict(torch.load(path1,map_location=torch.device('cpu')))
            if path2 is not None:
                self.Worker_Q_training.load_state_dict(torch.load(path2,map_location=torch.device('cpu')))
                self.Worker_Q_target.load_state_dict(torch.load(path2,map_location=torch.device('cpu')))
        else:
            if path1 is not None:
                self.Q_target.load_state_dict(torch.load(path1))
                self.Q_training.load_state_dict(torch.load(path1))
            if path2 is not None:
                self.Worker_Q_training.load_state_dict(torch.load(path2))
                self.Worker_Q_target.load_state_dict(torch.load(path2))


    def update_Qtarget(self,tau=0.005):
        for target_param, train_param in zip(self.Q_target.parameters(), self.Q_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)
        for target_param, train_param in zip(self.Worker_Q_target.parameters(), self.Worker_Q_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def train(self,batch_size=512,train_times=10):
        c_loss=[]
        a_loss=[]

        worker_loss = []

        pbar = tqdm.tqdm(range(train_times))
        torch.set_grad_enabled(True)
        self.Q_training.train()
        self.Worker_Q_training.train()
        self.Q_target.eval()
        self.Worker_Q_target.eval()
        for _ in pbar:
        # for _ in range(train_times):
            worker_state, order_state, order_num, new_order_state, \
                price_old, price_log_prob_old, reward, delta_t, \
                worker_state_next, order_state_next, order_num_next, new_order_state_next, \
                reservation_value, price_next, worker_action, worker_reward = self.buffer.sampling(batch_size, self.device)

            x1, x2, x3 = norm(new_order_state,worker_state,order_state)
            current_state_value, price_mu, price_sigma = self.Q_training(x1,x2,x3,order_num)
            current_worker_q_value = self.Worker_Q_training(torch.concat([x1,x2,reservation_value.unsqueeze(-1),price_old.unsqueeze(-1)],dim=-1), x3, order_num)
            current_worker_q_value = current_worker_q_value[torch.arange(current_worker_q_value.size(0)),worker_action]

            x1, x2, x3 = norm(new_order_state_next, worker_state_next, order_state_next)
            next_state_value, _, _ = self.Q_target(x1, x2, x3, order_num_next)
            # next_state_value, _, _ = self.Q_training(x1, x2, x3, order_num_next)
            next_worker_q_value = self.Worker_Q_target(torch.concat([x1, x2, reservation_value.unsqueeze(-1), price_next.unsqueeze(-1)], dim=-1), x3, order_num_next)

            # next_worker_q_value = torch.max(next_worker_q_value.detach(),dim=-1)[0]
            next_worker_q_value_index = torch.max(self.Worker_Q_training(torch.concat([x1, x2, reservation_value.unsqueeze(-1), price_next.unsqueeze(-1)], dim=-1), x3, order_num_next).detach(),dim=-1)[1]
            next_worker_q_value = next_worker_q_value[torch.arange(next_worker_q_value.size(0)), next_worker_q_value_index]


            worker_target = worker_reward + self.gamma** delta_t * next_worker_q_value
            worker_target = worker_target.float()

            td_target = reward + self.gamma ** delta_t * next_state_value.detach()
            td_target = td_target.float()

            critic_loss = self.loss_func(current_state_value, td_target)

            normal_dist = torch.distributions.Normal(price_mu, price_sigma)
            price_log_prob = normal_dist.log_prob(price_old)
            ratio = torch.exp(price_log_prob - price_log_prob_old)
            advantage = (td_target - current_state_value).detach() # currently, for simplify, only use one step td-error to approximate the advantage (may need to be changed)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            # train platform_net
            loss = critic_loss + actor_loss * 20
            self.optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Q_training.parameters(), 1.0)  # avoid gradient explosion

            has_nan = False
            for name, param in self.Q_training.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        break
            if has_nan:
                # print("NAN Gradient->Skip")
                continue

            self.optim.step()
            c_loss.append(critic_loss.item())
            a_loss.append(actor_loss.item())

            # train worker_net
            loss = self.loss_func(current_worker_q_value, worker_target)
            self.optim_worker.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Worker_Q_training.parameters(), 1.0)  # avoid gradient explosion

            has_nan = False
            for name, param in self.Worker_Q_training.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        break
            if has_nan:
                # print("NAN Gradient->Skip")
                continue

            self.optim_worker.step()
            worker_loss.append(loss.item())


        self.update_Qtarget()

        return np.mean(c_loss), np.mean(a_loss), np.mean(worker_loss)

'''
update state for each worker:

input:
current state: observe_space, current_orders, current_orders_num, positive_history, negative_history, speed, current_travel_route, current_travel_time
experience
action / guidance of next state: feedback, new_route ,new_route_time ,new_remaining_time ,new_total_travel_time
information about worker network: worker_feed_back, reservation_value

output:
new state: observe_space, current_orders, current_orders_num, positive_history, negative_history, current_travel_route, current_travel_time (speed is fixed, no need to return)
experience
full_experience: not None only if an experience is full filled
finished_order_time: not None only if any order is finished
worker_reward
'''
def single_update(observe_space, current_orders, current_orders_num, positive_history, negative_history, speed, capacity, current_travel_route, current_travel_time, experience, feedback, new_route ,new_route_time ,new_remaining_time ,new_total_travel_time, worker_feed_back, reservation_value):
    full_experience = None
    finished_order_time = None
    current_orders_num = int(current_orders_num)
    worker_reward = 0
    update_rate = 0.1
    # take action
    if feedback is not None:
        worker_reward=worker_feed_back[1]
        if feedback[-1] == -1: # -1 means reject
            negative_history = negative_history * update_rate + feedback[1][0] * (1-update_rate)
        else: # accept order
            positive_history = positive_history * update_rate + feedback[1][0] * (1-update_rate)

        # update experience
        if len(experience) > 0:
            experience.append(feedback[0][-1] - experience[0][-1]) # △t
            experience.append([feedback[0],speed,capacity,positive_history,negative_history]) # s_next
            experience.append([worker_feed_back,reservation_value]) # worker_next
            # print([speed,capacity,positive_history,negative_history])
            if len(experience) == 7:
                full_experience = experience
            else:
                print("There is a bug (experience)!!")
            experience = []
        experience.append([feedback[0],speed,capacity,positive_history,negative_history]) # s_current
        experience.append([worker_feed_back, reservation_value])  # worker_current
        experience.append(feedback[1]) # a
        experience.append(feedback[2]) # r

        # update state
        if feedback[-1] != -1: # accept order

            observe_space[2] -= 1 # remaining seat
            observe_space[3] = 1 # update to picking up state
            observe_space[4] = feedback[-1] # remaining picking up time
            observe_space[0] = feedback[0][-2][0] # plat
            observe_space[1] = feedback[0][-2][1] # plon

            current_travel_route, current_travel_time = new_route ,new_route_time

            if current_orders_num>0:
                current_orders[:current_orders_num,2], current_orders[:current_orders_num,3] = new_remaining_time[1:], new_total_travel_time[1:] # remaining travel time & total travel time (old orders)
            # print(new_remaining_time,new_total_travel_time,observe_space[4])
            current_orders[current_orders_num,2], current_orders[current_orders_num,3] = new_remaining_time[0], new_total_travel_time[0] # remaining travel time & total travel time (new orders)
            current_orders[current_orders_num, 0], current_orders[current_orders_num, 1] = feedback[0][-2][2], feedback[0][-2][3] # dlat,dlon (new orders)
            current_orders_num += 1

    step = speed * 1 # 1min
    if observe_space[3] == 1: # pick up
        if observe_space[4] > step:
            observe_space[4] -= step
        else: # finish picking up
            step -= observe_space[4]
            observe_space[4] = 0
            if observe_space[2] == 0: # no available seat
                observe_space[3] = 2
            else: # have available seat
                observe_space[3] = 0
    if step > 0 and current_orders_num != 0 :
        # go forward <step>
        step_minute = step
        step = step * 60
        for i in range(len(current_travel_time)):
            if step >= current_travel_time[i]:
                step -= current_travel_time[i]
            else:
                current_travel_time[i] -= step
                current_travel_time = current_travel_time[i:]
                current_travel_route = current_travel_route[i:]
                break
            if i == len(current_travel_time) - 1: # finish all orders
                observe_space[0], observe_space[1] = current_travel_route[-1][1], current_travel_route[-1][0]  # lat, lon
                current_travel_time = []
                current_travel_route = []

        # print(current_travel_route)
        if len(current_travel_route)>0:
            observe_space[0], observe_space[1] = current_travel_route[0][1], current_travel_route[0][0] # lat, lon

        current_orders[:current_orders_num, 2] -= step_minute # update remaining time

        # delete finished orders
        drop_index = np.zeros(current_orders.shape[0])
        drop_index[:current_orders_num] = (current_orders[:current_orders_num, 2] <= 0)
        drop_num = np.sum(drop_index)
        if drop_num>0:
            current_orders_num -= drop_num
            observe_space[2] += drop_num
            if observe_space[3] == 2:
                observe_space[3] = 0
            drop_index = drop_index.astype(bool)
            finished_orders = current_orders[drop_index]
            current_orders = current_orders[~drop_index]
            fill_matrix = np.zeros_like(finished_orders)
            current_orders = np.concatenate([current_orders,fill_matrix],axis=0)
            finished_order_time = finished_orders[:,3] / speed

    return observe_space, current_orders, current_orders_num, positive_history, negative_history, current_travel_route, current_travel_time, experience, full_experience, finished_order_time, worker_reward



if __name__ == '__main__':
    # test
    worker=Worker(1000)
    worker.reset()
    import matplotlib.pyplot as plt
    plt.plot(range(len(worker.positive_history)),worker.positive_history,'r')
    plt.plot(range(len(worker.positive_history)),worker.negative_history,'b')
    plt.show()
    # plot_accept_rate()



