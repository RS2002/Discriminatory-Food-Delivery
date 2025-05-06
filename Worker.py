import math
import numpy as np
import torch
import pandas as pd
from models import Worker_Q_Net, Assignment_Net
from joblib import Parallel, delayed
import torch.nn as nn
import tqdm
import warnings
import random
from sklearn.cluster import KMeans

# ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
INF = 1e8

def accept_rate(price=1.0, reservation_value=1.0):
    ratio = price / reservation_value
    return 1 / (1 + math.exp(-50 * (ratio - 0.95)))


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
def norm(order, x_state, x_order, lat_min=22.24370366972477, lat_max=22.505171559633027, lon_min=113.93901100917432,
         lon_max=114.26928623853212, wait_max_time=5, transportation_max_time=40, max_seat=3):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(order, torch.Tensor):
        order, x_state, x_order = order.clone(), x_state.clone(), x_order.clone()
    else:
        order, x_state, x_order = order.copy(), x_state.copy(), x_order.copy()

    # 1. lat & lon
    order[:, 0] = (order[:, 0] - lat_min) / lat_range
    order[:, 2] = (order[:, 2] - lat_min) / lat_range
    order[:, 1] = (order[:, 1] - lon_min) / lon_range
    order[:, 3] = (order[:, 3] - lon_min) / lon_range

    x_state[:, 0] = (x_state[:, 0] - lat_min) / lat_range
    x_state[:, 1] = (x_state[:, 1] - lon_min) / lon_range

    x_order[:, :, 0] = (x_order[:, :, 0] - lat_min) / lat_range * (x_order[:, :, 0] != 0)
    x_order[:, :, 1] = (x_order[:, :, 1] - lon_range) / lon_range * (x_order[:, :, 1] != 0)

    # 2. time
    order[:, 4] = order[:, 4] / wait_max_time  # max wait time: 5 min
    x_order[:, :, 2:4] = x_order[:, :, 2:4] / transportation_max_time  # max transportation time: 40min as threshold

    # 3. seat
    x_state[:, 2] = x_state[:, 2] / max_seat  # max seat: 3
    x_state[:, 4] = x_state[:, 4] / max_seat  # max seat: 3

    # 4. reservation_value
    x_state[:,5] = (x_state[:,5] - 0.85) / 0.3
    x_state[:,6] = (x_state[:,6] - 0.85) / 0.3
    x_order[:, :, 4] = (x_order[:, :, 4] - 0.85) / 0.3 * (x_order[:, :, 4] != 0)

    return order, x_state, x_order


def worker_state_norm(x_state, lat_min=22.24370366972477, lat_max=22.505171559633027, lon_min=113.93901100917432,
                      lon_max=114.26928623853212, max_seat=3):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(x_state, torch.Tensor):
        x_state = x_state.clone()
    else:
        x_state = x_state.copy()

    x_state[:, 0] = (x_state[:, 0] - lat_min) / lat_range
    x_state[:, 1] = (x_state[:, 1] - lon_min) / lon_range
    x_state[:, 2] = x_state[:, 2] / max_seat
    x_state[:, 4] = x_state[:, 4] / max_seat

    x_state[:,5] = (x_state[:,5] - 0.85) / 0.3
    x_state[:,6] = (x_state[:,6] - 0.85) / 0.3

    return x_state

def private_norm(x_state, lat_min=22.24370366972477, lat_max=22.505171559633027, lon_min=113.93901100917432,
                      lon_max=114.26928623853212, max_seat=3):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(x_state, torch.Tensor):
        x_state = x_state.clone()
    else:
        x_state = x_state.copy()

    x_state[:, 0] = (x_state[:, 0] - lat_min) / lat_range
    x_state[:, 1] = (x_state[:, 1] - lon_min) / lon_range
    x_state[:, 3] = x_state[:, 3] / max_seat

    x_state[:,4] = (x_state[:,4] - 0.85) / 0.3
    return x_state


class Buffer_Mask():
    def __init__(self, capacity=1e5):
        super().__init__()
        self.reset(capacity)

    def reset(self, capacity=None):
        if capacity is not None:
            self.capacity = capacity

        self.num = 0
        self.x = []
        self.x2 = []
        self.mask = []
        self.y = []
        self.episode = []

    def append(self, x, x2, mask, y, episode=0):
        self.x.extend(x.tolist())
        self.x2.extend(x2.tolist())
        self.mask.extend(mask.tolist())
        self.y.extend(y.tolist())
        self.episode.extend([episode] * len(x))
        self.num = len(self.x)

        if self.num > self.capacity:
            self.x = self.x[-self.capacity:]
            self.x2 = self.x2[-self.capacity:]
            self.mask = self.mask[-self.capacity:]
            self.y = self.y[-self.capacity:]
            self.episode = self.episode[-self.capacity:]
            self.num = self.capacity

    def sampling(self, size, device):
        # indices = np.random.randint(0, self.num, size=size)
        # indices = np.random.choice(self.num, size=size, replace=False)
        indices = np.random.choice(self.num, size=size)

        # priority = np.array(self.episode)
        # priority = priority - np.min(priority) + 1
        # probabilities = np.array(priority) / np.sum(priority)
        # indices = np.random.choice(self.num, size, p=probabilities, replace=False)

        x = torch.tensor([self.x[i] for i in indices]).to(device)
        x2 = torch.tensor([self.x2[i] for i in indices]).to(device)
        mask = torch.tensor([self.mask[i] for i in indices]).to(device)
        y = torch.tensor([self.y[i] for i in indices]).to(device)

        return x, x2, mask, y


# FIFO Buffer
class Buffer():
    def __init__(self, capacity=1e5):
        super().__init__()
        self.reset(capacity)

    def reset(self, capacity=None):
        if capacity is not None:
            self.capacity = capacity

        self.num = 0

        # state
        self.worker_state = []
        self.order_state = []
        self.order_num = []
        self.new_order_state = []

        # action
        self.price = []  # also worker extra state
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
        self.reservation_value = []  # worker extra state
        self.worker_action = []
        self.worker_reward = []
        self.price_next = []  # worker extra state (next)
        self.workload_current = []
        self.workload_next = []

        self.episode = []
        self.id = []

        self.mask = []

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

    def append(self, record, worker_id, mask, episode=1):
        episode_capacity = 10
        if self.num > 0 and self.episode[0] < episode - episode_capacity:
            episode_np = np.array(self.episode)
            old_record_num = len(episode_np[episode_np < (episode - episode_capacity)])
            self.num -= old_record_num
            self.worker_state = self.worker_state[old_record_num:]
            self.order_state = self.order_state[old_record_num:]
            self.order_num = self.order_num[old_record_num:]
            self.new_order_state = self.new_order_state[old_record_num:]
            self.price = self.price[old_record_num:]
            self.price_log_prob = self.price_log_prob[old_record_num:]
            self.delta_t = self.delta_t[old_record_num:]
            self.worker_state_next = self.worker_state_next[old_record_num:]
            self.order_state_next = self.order_state_next[old_record_num:]
            self.order_num_next = self.order_num_next[old_record_num:]
            self.new_order_state_next = self.new_order_state_next[old_record_num:]
            self.reward = self.reward[old_record_num:]
            self.reservation_value = self.reservation_value[old_record_num:]
            self.worker_action = self.worker_action[old_record_num:]
            self.worker_reward = self.worker_reward[old_record_num:]
            self.price_next = self.price_next[old_record_num:]
            self.workload_current = self.workload_current[old_record_num:]
            self.workload_next = self.workload_next[old_record_num:]
            self.episode = self.episode[old_record_num:]
            self.id = self.id[old_record_num:]
            self.mask = self.mask[old_record_num:]
            if self.episode[0] < episode - episode_capacity:
                print("Buffer Error!")
                exit(-1)

        state, worker_current, action, reward, delta_t, next_state, worker_next = record

        speed, capacity, positive, negative, time = state[1], state[2], state[3], state[4], state[5]
        state = state[0]

        speed_next, capacity_next, positive_next, negative_next, time_next = next_state[1], next_state[2], next_state[
            3], next_state[4], next_state[5]
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
            self.workload_current = self.workload_current[1:]
            self.workload_next = self.workload_next[1:]
            self.episode = self.episode[1:]
            self.id = self.id[1:]
            self.mask = self.mask[1:]
        else:
            self.num += 1

        worker_state_temp = state[0][:3].tolist()
        worker_state_temp.extend([speed, capacity, positive, negative, time / 60])
        self.worker_state.append(worker_state_temp)
        self.order_state.append(state[1].tolist())
        self.order_num.append(state[2])
        self.new_order_state.append(state[3].tolist())
        self.price.append(action[0])
        self.price_log_prob.append(action[1])
        self.delta_t.append(delta_t)
        worker_state_next_temp = next_state[0][:3].tolist()
        worker_state_next_temp.extend([speed_next, capacity_next, positive_next, negative_next, time_next / 60])
        self.worker_state_next.append(worker_state_next_temp)
        self.order_state_next.append(next_state[1].tolist())
        self.order_num_next.append(next_state[2])
        self.new_order_state_next.append(next_state[3].tolist())
        self.reward.append(reward)
        self.reservation_value.append(worker_current[1])
        self.worker_action.append(worker_current[0][0])
        self.worker_reward.append(worker_current[0][1])
        self.price_next.append(worker_next[0][2])
        self.workload_current.append(worker_current[0][3])
        self.workload_next.append(worker_next[0][3])

        self.episode.append(episode)
        self.id.append(worker_id)
        self.mask.append(mask)

    '''
    random sample <size> samples
    return:

    line1: platform & worker common state
    line2: action (price is also a part of satet of worker), reward, △t
    line3: platform & worker common state_next
    line4: worker extra state/action/reward/state_next
    '''

    def sampling(self, size, device):
        # indices = np.random.randint(0, self.num, size=size)
        # indices = np.random.choice(self.num, size=size, replace=False)
        indices = np.random.choice(self.num, size=size)

        # priority = np.array(self.episode)
        # priority = priority - np.min(priority) + 1
        # probabilities = np.array(priority) / np.sum(priority)
        # indices = np.random.choice(self.num, size, p=probabilities)

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
        workload_current = torch.tensor([self.workload_current[i] for i in indices]).to(device)
        workload_next = torch.tensor([self.workload_next[i] for i in indices]).to(device)
        mask = torch.tensor([self.mask[i] for i in indices]).to(device)

        return worker_state, order_state, order_num, new_order_state, \
            price, price_log_prob, reward, delta_t, \
            worker_state_next, order_state_next, order_num_next, new_order_state_next, \
            reservation_value, price_next, worker_action, worker_reward, workload_current, workload_next, mask

    def sample_episode(self, current_episode, device):
        indices = np.where(np.array(self.episode) == current_episode)[0]
        indices = np.sort(indices)

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
        workload_current = torch.tensor([self.workload_current[i] for i in indices]).to(device)
        workload_next = torch.tensor([self.workload_next[i] for i in indices]).to(device)

        id = torch.tensor([self.id[i] for i in indices]).to(device)
        mask = torch.tensor([self.mask[i] for i in indices]).to(device)

        return worker_state, order_state, order_num, new_order_state, \
            price, price_log_prob, reward, delta_t, \
            worker_state_next, order_state_next, order_num_next, new_order_state_next, \
            reservation_value, price_next, worker_action, worker_reward, workload_current, workload_next, id, mask


'''
num: worker number
history_num: the number of history positive and negative unit-price for each worker
reservation_value/speed: 1.0 as baseline
capacity: the maximum order number of each worker
'''


class Worker():
    def __init__(self, buffer, buffer_price, buffer_mask, lr=0.0001, gamma=0.99, eps_clip=0.2, max_step=60,
                 history_num=5, num=1000, reservation_value=None, speed=None, capacity=None, group=None, device=None,
                 zone_table_path="../data/zone_table.csv", model_path=None,
                 njobs=24, intelligent_worker=False, probability_worker=False, bilstm=False, dropout=0.0, pretrain = False, worker_mode = "gdpr"):
        super().__init__()
        self.intelligent_worker = intelligent_worker
        self.probability_worker = probability_worker
        self.pretrain = pretrain

        self.buffer_q = buffer
        self.buffer_price = buffer_price
        self.buffer = self.buffer_q

        self.buffer_mask = buffer_mask

        self.gamma = gamma

        # self.worker_gamma = gamma
        self.worker_gamma = 1

        self.eps_clip = eps_clip
        self.device = device
        self.history_num = history_num

        self.max_step = max_step

        self.zone_lookup = pd.read_csv(zone_table_path)
        self.coordinate_lookup = np.array(self.zone_lookup[['lat', 'lon']])

        self.Q_training = Assignment_Net(state_size=8, history_order_size=5, current_order_size=17, hidden_dim=64, head=3,
                                bi_direction=bilstm, dropout=dropout).to(device)
        self.Q_target = Assignment_Net(state_size=8, history_order_size=5, current_order_size=17, hidden_dim=64, head=3,
                              bi_direction=bilstm, dropout=dropout).to(device)

        if self.intelligent_worker:
            self.Worker_Q_training = Worker_Q_Net(input_size=15, history_order_size=5, output_dim=2,
                                                  bi_direction=bilstm, dropout=dropout).to(device)
            self.Worker_Q_target = Worker_Q_Net(input_size=15, history_order_size=5, output_dim=2, bi_direction=bilstm,
                                                dropout=dropout).to(device)

        self.load(model_path, self.device)
        for param in self.Q_target.parameters():
            param.requires_grad = False
        self.Q_target.eval()

        print('Platform total parameters:', sum(
            p.numel() for p in self.Q_training.parameters() if
            p.requires_grad))

        if self.intelligent_worker:
            for param in self.Worker_Q_target.parameters():
                param.requires_grad = False
            self.Worker_Q_target.eval()
            print('Worker total parameters:',
                  sum(p.numel() for p in self.Worker_Q_training.parameters() if p.requires_grad))
            self.optim_worker = torch.optim.Adam(self.Worker_Q_training.parameters(), lr=lr, weight_decay=0.00)
            self.schedule_worker = torch.optim.lr_scheduler.ExponentialLR(self.optim_worker, gamma=0.99)

        self.update_Qtarget(tau=1.0)

        self.loss_func = nn.MSELoss(reduction="none")
        # self.loss_func = nn.MSELoss()

        # self.optim = torch.optim.Adam(self.Q_training.parameters(), lr=lr, weight_decay=0.01)
        self.optim = torch.optim.Adam(self.Q_training.parameters(), lr=lr)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        self.optim2 = torch.optim.Adam(self.Q_training.parameters(), lr=lr * 0.1)
        self.schedule2 = torch.optim.lr_scheduler.ExponentialLR(self.optim2, gamma=0.97)
        # self.reset(max_step,num,reservation_value, speed, capacity, group)

        self.njobs = njobs
        self.max_norm = 1.0
        self.worker_mode = worker_mode

    def reset(self, max_step=60, num=1000, reservation_value=None, speed=None, capacity=None, group=None, train=True, demand_sample_rate = 0.2, mask_rate = None):
        torch.set_grad_enabled(False)

        self.time = 0

        if train:
            if self.intelligent_worker:
                self.Worker_Q_training.train()
            self.Q_training.train()
        else:
            if self.intelligent_worker:
                self.Worker_Q_training.eval()
            self.Q_training.eval()

        self.is_train = train
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

        if self.probability_worker:
            self.positive_history = np.zeros([self.num, self.history_num])
            self.negative_history = np.zeros([self.num, self.history_num])
            for i in range(self.num):
                index_pos = 0
                index_neg = 0
                while index_neg < self.history_num or index_pos < self.history_num:
                    record = np.random.randn((5 * self.history_num))
                    record = np.abs(record * 0.01 + self.reservation_value[i])
                    rand = np.random.rand((5 * self.history_num))
                    for j in range(5 * self.history_num):
                        acc_rate = accept_rate(record[j], self.reservation_value[i])
                        if rand[j] <= acc_rate and index_pos < self.history_num:
                            self.positive_history[i, index_pos] = record[j]
                            index_pos += 1
                        elif rand[j] > acc_rate and index_neg < self.history_num:
                            self.negative_history[i, index_neg] = record[j]
                            index_neg += 1
                        if index_pos >= self.history_num and index_pos >= self.history_num:
                            break
            '''
            use single EMA to replace history record (reduce state space size)
            '''
            self.positive_history = np.mean(self.positive_history, axis=-1)
            self.negative_history = np.mean(self.negative_history, axis=-1)
        else:
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
        self.observe_space = np.zeros([self.num, 5])
        self.observe_space[:, 2] = self.capacity

        # allocate a initial location randomly from valid zone
        random_integers = np.random.randint(0, len(self.zone_lookup), size=(self.num))
        self.observe_space[:, :2] = self.coordinate_lookup[random_integers]

        '''
        current orders
        0,1: drop-off lat,lon
        2: remaining transportation time (approximated)
        3: total transportation time (approximated)
        4: unit price
        '''
        self.current_orders = np.zeros([self.num, int(self.max_capacity), 5])
        self.current_order_num = np.zeros([self.num])

        # some records for simulation
        self.travel_route = [[] for _ in range(self.num)]
        self.travel_time = [[] for _ in range(self.num)]
        self.experience = [[] for _ in range(self.num)]  # When each item gets full, it will be added to buffer.
        self.Pass_Travel_Time = []

        self.price = [[] for _ in range(self.num)]
        self.salary = [0.0] * self.num
        self.work_load = [0.0] * self.num
        self.worker_assign_order = [0] * self.num
        self.worker_reject_order = [0] * self.num

        if mask_rate is not None:
            self.gen_mask_eval(mask_rate)
        else:
            if self.pretrain:
                if train:
                    self.gen_mask_pretrain()
                else:
                    self.gen_mask_eval(0.5)
            else:
                self.gen_mask()

        self.demand_sample_rate = demand_sample_rate


        self.avg = None

        self.mask_prop = torch.mean((self.mask == 1).float()).item()
        self.strike_prop = torch.mean((self.mask == 2).float()).item()

        self.global_state = np.zeros([12])
        self.global_state[-1] = demand_sample_rate
        self.global_state[-2] = torch.sum((self.mask == 1).float()).item() / 1000
        mask = self.mask.cpu().numpy()
        reservation_value_hat = (self.positive_history + self.negative_history) / 2
        for i in range(10):
            if i == 0:
                mask_temp = mask[reservation_value_hat < 0.88]
            elif i == 9:
                mask_temp = mask[reservation_value_hat >= 1.12]
            else:
                mask_temp = mask[(reservation_value_hat >= 0.85 + 0.03 * i) & (reservation_value_hat < 0.88 + 0.03 * i)]

            self.global_state[i] = np.sum((mask_temp == 0)) / 100

    def reset_mask(self,mask_rate=None,type=None):
        if mask_rate is None:
            self.gen_mask(type)
        else:
            self.gen_mask_eval(mask_rate)

        self.avg = None

        self.mask_prop = torch.mean((self.mask == 1).float()).item()
        self.strike_prop = torch.mean((self.mask == 2).float()).item()

        self.global_state = np.zeros([12])
        self.global_state[-1] = self.demand_sample_rate
        self.global_state[-2] = torch.sum((self.mask == 1).float()).item() / 1000
        mask = self.mask.cpu().numpy()
        reservation_value_hat = (self.positive_history + self.negative_history) / 2
        for i in range(10):
            if i == 0:
                mask_temp = mask[reservation_value_hat < 0.88]
            elif i == 9:
                mask_temp = mask[reservation_value_hat >= 1.12]
            else:
                mask_temp = mask[(reservation_value_hat >= 0.85 + 0.03 * i) & (reservation_value_hat < 0.88 + 0.03 * i)]

            self.global_state[i] = np.sum((mask_temp == 0)) / 100

    def gen_mask(self,type=None):
        if type is None:
            type = self.worker_mode

        t = np.array([[0]] * self.num)
        worker_state = np.concatenate(
            [self.observe_space[:, :3], np.expand_dims(self.speed, axis=-1), np.expand_dims(self.capacity, axis=-1),
             np.expand_dims(self.positive_history, axis=-1), np.expand_dims(self.negative_history, axis=-1), t],
            axis=-1)
        x = worker_state_norm(worker_state)
        self.x = x
        x = torch.from_numpy(x).float().to(self.device)

        x2 = np.concatenate(
            [self.observe_space[:, :2], np.expand_dims(self.speed, axis=-1), np.expand_dims(self.capacity, axis=-1),
             np.expand_dims(self.reservation_value, axis=-1)],
            axis=-1)
        x2 = private_norm(x2)
        self.x2 = x2
        x2 = torch.from_numpy(x2).float().to(self.device)

        y = self.Q_training.estimate_earning(x,x2)
        self.y = y.cpu().numpy()
        samples = torch.normal(y[..., 0], y[..., 1])

        if type == "benchmark":
            samples[:,1] = -INF
        elif type == "mask":
            samples[:,0] = -INF

        self.mask = torch.argmax(samples, dim=-1)

    def mask_append(self, lowest_utility=35, episode=0):
        x = self.x
        x2 = self.x2
        mask = self.mask.cpu().numpy()
        # y = self.worker_reward.copy()
        y = self.worker_reward
        y[mask == 2] = lowest_utility
        self.lowest_utility = lowest_utility
        self.buffer_mask.append(x, x2, mask, y, episode)

    def calculate_courier_loss(self, lowest_utility=35):
        y = self.worker_reward
        mask = self.mask.cpu().numpy()
        y[mask == 2] = lowest_utility
        y_hat = self.y[np.arange(y.shape[0]), mask, 0]
        mse = np.mean((y-y_hat)**2)
        mape = np.mean(np.abs(y-y_hat)/(y+1e-5))
        return mse, mape

    def gen_mask_eval(self, mask_rate=0.5):
        self.mask = (torch.rand([self.num]) < mask_rate).int().to(self.device)

    def gen_mask_pretrain(self):
        rand = random.random()
        group_num = random.randint(5, 30)
        t = np.array([[0]] * self.num)

        if rand < 0.1:  # random mask
            rand = np.random.rand(self.num)
            mask_prob = random.random()
            mask = (rand < mask_prob).astype(np.int32)
        elif rand < 0.2:  # random strike
            rand = np.random.rand(self.num)
            strike_prob = random.random()
            mask = (rand < strike_prob).astype(np.int32)
            mask *= 2
        elif rand < 0.3:  # random mask + strike
            rand = np.random.rand(self.num)
            mask_prob = random.random()
            strike_prob = random.random()

            mask_prob *= 0.5
            strike_prob *= 0.5

            mask = (rand < (mask_prob + strike_prob)).astype(np.int32)
            mask[rand < strike_prob] = 2
        else:  # divide into groups first
            if rand < 0.6:
                model = self.Q_training.worker_encode
                worker_state = np.concatenate(
                    [self.observe_space[:, :3], np.expand_dims(self.speed, axis=-1),
                     np.expand_dims(self.capacity, axis=-1),
                     np.expand_dims(self.positive_history, axis=-1), np.expand_dims(self.negative_history, axis=-1), t],
                    axis=-1)
                x = worker_state_norm(worker_state)
                x = torch.from_numpy(x).float().to(self.device)
                y = model(x)
                y = y.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=group_num)
                kmeans.fit(y)
                labels = kmeans.labels_
            else:  # cluster by reservation value
                sorted_data = np.sort(self.reservation_value)
                cut_points = np.random.choice(np.arange(1, self.num), group_num - 1, replace=False)
                cut_points.sort()
                labels = np.zeros(self.num, dtype=int)
                for i in range(group_num - 1):
                    labels[self.reservation_value >= sorted_data[cut_points[i]]] = i + 1

            rand = random.random()
            if rand < 0.5: # random label
                rand = random.random()
                if rand < 0.3:  # only mask
                    scale_rate = random.random() * 2
                    group_mask_prob = np.random.rand(group_num)
                    mask_prob = group_mask_prob[labels] * scale_rate
                    rand = np.random.rand(self.num)
                    mask = (rand < mask_prob).astype(np.int32)
                elif rand < 0.6:  # only strike
                    scale_rate = random.random()
                    group_mask_prob = np.random.rand(group_num)
                    strike_prob = group_mask_prob[labels] * scale_rate
                    rand = np.random.rand(self.num)
                    mask = (rand < strike_prob).astype(np.int32)
                    mask *= 2
                else:  # mask + strike
                    scale_rate1 = random.random() * 2
                    group_mask_prob = np.random.rand(group_num)
                    mask_prob = group_mask_prob[labels] * scale_rate1

                    scale_rate2 = random.random()
                    group_mask_prob = np.random.rand(group_num)
                    strike_prob = group_mask_prob[labels] * scale_rate2

                    mask_prob *= 0.5
                    strike_prob *= 0.5

                    rand = np.random.rand(self.num)
                    mask = (rand < (mask_prob + strike_prob)).astype(np.int32)
                    mask[rand < strike_prob] = 2
            else: # monotonous label
                rand = random.random()
                if rand < 0.3:  # only mask
                    scale_rate = random.random() * 2
                    group_mask_prob = np.random.rand(group_num)

                    group_mask_prob = np.sort(group_mask_prob)
                    if random.random() < 0.5:
                        group_mask_prob = group_mask_prob[::-1]

                    mask_prob = group_mask_prob[labels] * scale_rate
                    rand = np.random.rand(self.num)
                    mask = (rand < mask_prob).astype(np.int32)
                elif rand < 0.6:  # only strike
                    scale_rate = random.random()
                    group_mask_prob = np.random.rand(group_num)

                    group_mask_prob = np.sort(group_mask_prob)
                    if random.random() < 0.5:
                        group_mask_prob = group_mask_prob[::-1]

                    strike_prob = group_mask_prob[labels] * scale_rate
                    rand = np.random.rand(self.num)
                    mask = (rand < strike_prob).astype(np.int32)
                    mask *= 2
                else:  # mask + strike
                    scale_rate1 = random.random() * 2
                    group_mask_prob = np.random.rand(group_num)

                    group_mask_prob = np.sort(group_mask_prob)
                    if random.random() < 0.5:
                        group_mask_prob = group_mask_prob[::-1]

                    mask_prob = group_mask_prob[labels] * scale_rate1

                    scale_rate2 = random.random()
                    group_mask_prob = np.random.rand(group_num)

                    group_mask_prob = np.sort(group_mask_prob)
                    if random.random() < 0.5:
                        group_mask_prob = group_mask_prob[::-1]

                    strike_prob = group_mask_prob[labels] * scale_rate2

                    mask_prob *= 0.5
                    strike_prob *= 0.5

                    rand = np.random.rand(self.num)
                    mask = (rand < (mask_prob + strike_prob)).astype(np.int32)
                    mask[rand < strike_prob] = 2

        self.mask = torch.from_numpy(mask).to(self.device)

        if self.worker_mode == "benchmark":
            self.mask[self.mask==1]=0
        elif self.worker_mode == "mask":
            self.mask[self.mask==0]=1


    def observe(self, order, current_time, exploration_rate=0):
        self.time = current_time
        t = np.array([[self.time / 60]] * self.num)
        # self.Q_training.eval()
        # 1. contstruct the worker state
        # print(self.observe_space.shape, self.speed.shape, self.capacity.shape, self.positive_history.shape, self.negative_history.shape)
        worker_state = np.concatenate(
            [self.observe_space[:, :3], np.expand_dims(self.speed, axis=-1), np.expand_dims(self.capacity, axis=-1),
             np.expand_dims(self.positive_history, axis=-1), np.expand_dims(self.negative_history, axis=-1), t],
            axis=-1)
        # 2. construct the order state
        order_state = np.array(order[['plat', 'plon', 'dlat', 'dlon', 'minute']])
        order_state[:, -1] = current_time - order_state[:, -1]  # waiting time

        # m = np.array([[self.mask_prop]]*order_state.shape[0])
        # s = np.array([[self.strike_prop]]*order_state.shape[0])
        # order_state = np.concatenate([order_state, m, s],axis=-1)
        global_state = np.tile(self.global_state, (order_state.shape[0], 1))
        order_state = np.concatenate([order_state, global_state], axis=-1)

        # 3. get Q value
        x1, x2, x3 = norm(torch.from_numpy(order_state).to(self.device), torch.from_numpy(worker_state).to(self.device),
                          torch.from_numpy(self.current_orders).to(self.device))
        q_value, price_mu, price_sigma = self.Q_training(x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device), self.mask)

        q_value = (q_value[0]+q_value[1])/2

        exploration_matrix = torch.rand_like(q_value)
        q_value[exploration_matrix < exploration_rate] = INF
        # 4. delete the Q value of not available workers
        q_value[self.observe_space[:, 3] != 0] = -INF
        # 5. delete striking workers
        q_value[self.mask == 2] = -INF
        return q_value.cpu().detach().numpy(), price_mu.cpu().detach().numpy(), price_sigma.cpu().detach().numpy(), order_state, worker_state

    def update(self, feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
               new_total_travel_time_table, worker_feed_back_table, current_time, final_step=False, episode=1):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.observe_space[i], self.current_orders[i], self.current_order_num[i],
                                   self.positive_history[i], self.negative_history[i], self.speed[i], self.capacity[i],
                                   self.travel_route[i], self.travel_time[i], self.experience[i], feedback_table[i],
                                   new_route_table[i], new_route_time_table[i], new_remaining_time_table[i],
                                   new_total_travel_time_table[i], worker_feed_back_table[i], self.reservation_value[i],
                                   self.time)
            for i in range(self.num))

        for i in range(len(results)):
            # take some record
            if feedback_table[i] is not None:  # assign new order
                self.worker_assign_order[i] += 1
                if feedback_table[i][-1] == -1:  # reject order
                    self.worker_reject_order[i] += 1
                else:
                    price = feedback_table[i][1][0]
                    work_load = feedback_table[i][1][2]
                    salary = feedback_table[i][1][3]
                    self.work_load[i] += work_load
                    self.salary[i] += salary
                    self.price[i].append(price)

            # update state
            self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.positive_history[i], \
            self.negative_history[i], self.travel_route[i], self.travel_time[i], self.experience[i] \
                = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][
                6], results[i][7]
            if self.is_train and results[i][8] is not None:
                self.buffer.append(results[i][8], i, self.mask[i], episode)
            if results[i][9] is not None:
                self.Pass_Travel_Time.extend(results[i][9].tolist())
            self.worker_reward[i] += self.worker_gamma ** current_time * results[i][10]

        # take the ending into consideration
        if final_step and self.is_train:
            for i in range(self.num):
                if len(self.experience[i]) > 0:
                    self.experience[i].append(-1)  # △t: -1 represents done
                    self.experience[i].append(self.experience[i][0])  # meaningless: only used to keep a same dimension
                    self.experience[i].append(self.experience[i][1])
                    if len(self.experience[i]) == 7:
                        self.buffer.append(self.experience[i], i, self.mask[i], episode)
                    else:
                        print("There is a bug (final experience)!!")
            # finished_order_time = self.current_orders[i, :, 3]
            # finished_order_time = finished_order_time[finished_order_time!=0]
            # self.Pass_Travel_Time.extend(finished_order_time.tolist())

    def save(self, path1):
        torch.save(self.Q_training.state_dict(), path1)

    def load(self, path1=None, device=torch.device("cpu")):
        if path1 is not None:
            self.Q_target.load_state_dict(torch.load(path1), map_location=device, strict=False)
            self.Q_training.load_state_dict(torch.load(path1), map_location=device, strict=False)

    def update_Qtarget(self, tau=0.005):
        for target_param, train_param in zip(self.Q_target.parameters(), self.Q_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def train_actor(self, episode, batch_size=512, train_times=10, update_critic=False, lamada=0.9, kl_threshold=0.05):
        rate_entro = 0.05
        rate_kl = 3.0

        c_loss = []
        a_loss = []

        torch.set_grad_enabled(False)
        self.Q_training.eval()

        worker_state, order_state, order_num, new_order_state, \
            price_old, price_log_prob_old, reward, delta_t, \
            worker_state_next, order_state_next, order_num_next, new_order_state_next, \
            reservation_value, price_next, worker_action, worker_reward, workload_current, workload_next, worker_id, mask = self.buffer.sample_episode(
            episode, self.device)

        # print(new_order_state.shape, worker_state.shape, order_state.shape)

        # first calculate the advantage for PPO actor
        x1, x2, x3 = norm(new_order_state, worker_state, order_state)
        current_state_value_target, mu_old, sigma_old = self.Q_training(x1, x2, x3, order_num, mask)
        current_state_value_target = (current_state_value_target[0] + current_state_value_target[1]) / 2
        current_state_value_target = torch.diag(current_state_value_target).detach()
        mu_old, sigma_old = torch.diag(mu_old).detach(), torch.diag(sigma_old).detach()

        x1, x2, x3 = norm(new_order_state_next, worker_state_next, order_state_next)
        next_state_value_target, _, _ = self.Q_training(x1, x2, x3, order_num_next, mask)
        next_state_value_target = (next_state_value_target[0] + next_state_value_target[1]) / 2
        next_state_value_target = torch.diag(next_state_value_target).detach()

        next_state_value_target2, _, _ = self.Q_target(x1, x2, x3, order_num_next, mask)
        next_state_value_target2 = (next_state_value_target2[0] + next_state_value_target2[1]) / 2
        next_state_value_target2 = torch.diag(next_state_value_target2).detach()

        is_done = (delta_t == -1).float()
        td_target_target = reward + (self.gamma ** delta_t * next_state_value_target) * (1 - is_done)
        td_delta_target = td_target_target - current_state_value_target
        advantage_target = calculate_advantage(td_delta_target, delta_t, worker_id, gamma=self.gamma, lamada=lamada)

        td_target_target2 = reward + (self.gamma ** delta_t * next_state_value_target2) * (1 - is_done)
        # td_target_target = torch.min(td_target_target, td_target_target2)
        td_target_target = td_target_target2

        pbar = tqdm.tqdm(range(train_times))
        params = {k: v.clone() for k, v in self.Q_training.state_dict().items()}

        advantage = advantage_target
        td_target = td_target_target

        for m in pbar:

            torch.set_grad_enabled(False)
            self.Q_training.eval()

            x1, x2, x3 = norm(new_order_state, worker_state, order_state)
            _, mu_new, sigma_new = self.Q_training(x1, x2, x3, order_num, mask)
            mu_new, sigma_new = torch.diag(mu_new).detach(), torch.diag(sigma_new).detach()
            kl_div = kl_divergence(mu_new, sigma_new, mu_old, sigma_old)

            if m != 0:

                if kl_div > kl_threshold:
                    print("Big KL Divergence: ", kl_div)
                    rate_kl *= 2
                    if self.pretrain:
                        self.Q_training.load_state_dict(params)
                        if m == train_times - 1:
                            break
                else:
                    params = {k: v.clone() for k, v in self.Q_training.state_dict().items()}

                if update_critic:
                    current_state_value, _, _ = self.Q_training(x1, x2, x3, order_num, mask)
                    current_state_value = (current_state_value[0] + current_state_value[1]) / 2
                    current_state_value = torch.diag(current_state_value).detach()
                    x1, x2, x3 = norm(new_order_state_next, worker_state_next, order_state_next)
                    next_state_value, _, _ = self.Q_training(x1, x2, x3, order_num_next, mask)
                    next_state_value = (next_state_value[0] + next_state_value[1]) / 2
                    next_state_value = torch.diag(next_state_value).detach()

                    is_done = (delta_t == -1).float()
                    td_target = reward + (self.gamma ** delta_t * next_state_value) * (1 - is_done)
                    td_delta = td_target - current_state_value
                    advantage = calculate_advantage(td_delta, delta_t, worker_id, gamma=self.gamma, lamada=lamada)

                    advantage = 0.5 * advantage_target + 0.5 * advantage
                    # td_target = torch.min(td_target, td_target_target)

                    td_target = td_target_target
                    advantage_target = advantage
                    td_target_target = td_target

            torch.set_grad_enabled(True)
            self.Q_training.train()

            for _ in range(1):
                indices = torch.randint(0, advantage.shape[0], size=(int(batch_size),))
                worker_state_temp, order_state_temp, order_num_temp, new_order_state_temp, \
                    price_old_temp, price_log_prob_old_temp, reward_temp, delta_t_temp, \
                    worker_state_next_temp, order_state_next_temp, order_num_next_temp, new_order_state_next_temp, \
                    reservation_value_temp, price_next_temp, worker_action_temp, worker_reward_temp, workload_current_temp, workload_next_temp, td_target_temp, advantage_temp, mu_old_temp, sigma_old_temp, mask_temp = \
                    worker_state[indices], order_state[indices], order_num[indices], new_order_state[indices], \
                        price_old[indices], price_log_prob_old[indices], reward[indices], delta_t[indices], \
                        worker_state_next[indices], order_state_next[indices], order_num_next[indices], \
                    new_order_state_next[indices], \
                        reservation_value[indices], price_next[indices], worker_action[indices], worker_reward[indices], \
                    workload_current[indices], workload_next[indices], td_target[indices], advantage[indices], mu_old[
                        indices], sigma_old[indices], mask[indices]

                x1, x2, x3 = norm(new_order_state_temp, worker_state_temp, order_state_temp)
                current_state_value, price_mu, price_sigma = self.Q_training(x1, x2, x3, order_num_temp, mask_temp)
                # entropy_loss = gaussian_entropy(price_sigma)
                current_state_value = (current_state_value[0] + current_state_value[1]) / 2
                current_state_value, price_mu, price_sigma = torch.diag(current_state_value), torch.diag(
                    price_mu), torch.diag(price_sigma)

                # kl_loss = kl_divergence(price_mu, price_sigma, mu_old_temp, sigma_old_temp)
                kl_loss = kl_div
                entropy_loss = gaussian_entropy(price_sigma)

                weight_class = torch.ones([2]).to(self.device)
                for j in range(2):
                    weight_class[j] += torch.sum((mask_temp == j).float())
                weight_class = 1 / weight_class
                weight = weight_class[mask_temp]
                weight = weight * batch_size / 2

                if update_critic:
                    td_target_temp = td_target_temp.float()
                    critic_loss = self.loss_func(current_state_value, td_target_temp.detach())
                    critic_loss = critic_loss * weight
                    critic_loss = torch.mean(critic_loss)
                else:
                    critic_loss = 0

                normal_dist = torch.distributions.Normal(price_mu, price_sigma)
                price_log_prob = normal_dist.log_prob(price_old_temp)
                ratio = torch.exp(price_log_prob - price_log_prob_old_temp)

                surr1 = ratio * advantage_temp.detach()
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_temp.detach()

                # actor_loss = torch.mean(-torch.min(surr1, surr2))

                actor_loss = -torch.min(surr1, surr2)
                actor_loss = actor_loss * weight
                actor_loss = torch.mean(actor_loss)

                loss1 = actor_loss + rate_entro * entropy_loss + rate_kl * kl_loss
                loss2 = critic_loss

                if update_critic:
                    loss = loss1 + loss2
                else:
                    loss = loss1

                # self.optim.zero_grad()
                self.optim2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Q_training.parameters(), self.max_norm)
                has_nan = False
                for name, param in self.Q_training.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            has_nan = True
                            break
                if has_nan:
                    # print("NAN Gradient->Skip")
                    continue
                # self.optim.step()
                self.optim2.step()
                if update_critic:
                    c_loss.append(critic_loss.item())
                a_loss.append(actor_loss.item())

        if update_critic:
            self.update_Qtarget()
        else:
            c_loss.append(0)
        # self.schedule.step()
        # self.schedule2.step()
        torch.set_grad_enabled(False)
        return np.mean(c_loss), np.mean(a_loss)

    def train_critic(self, batch_size=512, train_times=30, show_pbar=False):
        c_loss = []
        a_loss = [0]

        if show_pbar:
            pbar = tqdm.tqdm(range(train_times))
        else:
            pbar = range(train_times)

        torch.set_grad_enabled(True)
        self.Q_training.train()

        for _ in pbar:
            worker_state, order_state, order_num, new_order_state, \
                price_old, price_log_prob_old, reward, delta_t, \
                worker_state_next, order_state_next, order_num_next, new_order_state_next, \
                reservation_value, price_next, worker_action, worker_reward, workload_current, workload_next, mask = self.buffer.sampling(
                batch_size, self.device)

            x1, x2, x3 = norm(new_order_state, worker_state, order_state)
            current_state_value, _, _ = self.Q_training(x1, x2, x3, order_num, mask)
            current_state_value1, current_state_value2 = current_state_value[0], current_state_value[1]
            current_state_value1, current_state_value2 = torch.diag(current_state_value1), torch.diag(current_state_value2)

            x1, x2, x3 = norm(new_order_state_next, worker_state_next, order_state_next)
            # next_state_value1, _, _ = self.Q_target(x1, x2, x3, order_num_next, mask)
            # next_state_value2, _, _ = self.Q_training(x1, x2, x3, order_num_next, mask)
            # next_state_value1, next_state_value2 = torch.diag(next_state_value1), torch.diag(next_state_value2)
            # next_state_value = torch.min(next_state_value1, next_state_value2)
            next_state_value, _, _ = self.Q_target(x1, x2, x3, order_num_next, mask)
            next_state_value1, next_state_value2 = next_state_value[0], next_state_value[1]
            next_state_value1, next_state_value2 = torch.diag(next_state_value1), torch.diag(next_state_value2)

            is_done = (delta_t == -1).float()
            td_target1 = reward + self.gamma ** delta_t * next_state_value1.detach() * (1 - is_done)
            td_target1 = td_target1.float()
            td_target2 = reward + self.gamma ** delta_t * next_state_value2.detach() * (1 - is_done)
            td_target2 = td_target2.float()

            critic_loss = self.loss_func(current_state_value1, td_target2.detach()) + self.loss_func(current_state_value2, td_target1.detach())
            # loss = critic_loss

            weight_class = torch.ones([2]).to(self.device)
            for j in range(2):
                weight_class[j] += torch.sum((mask==j).float())
            weight_class = 1 / weight_class
            weight = weight_class[mask]
            weight = weight * batch_size / 2
            critic_loss = critic_loss * weight
            critic_loss = torch.mean(critic_loss)
            loss = critic_loss

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q_training.parameters(), self.max_norm)

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

        self.update_Qtarget()
        # self.schedule.step()
        torch.set_grad_enabled(False)
        return np.mean(c_loss), np.mean(a_loss)

    def train_mask(self, batch_size=512, train_times=10):
        pbar = tqdm.tqdm(range(train_times))
        torch.set_grad_enabled(True)
        self.Q_training.train()
        loss_list = []

        scale = 0.01

        for _ in pbar:
            x, x2, mask, r = self.buffer_mask.sampling(batch_size, self.device)
            x, x2, mask, r = x.float(), x2.float(), mask.int(), r.float()
            y = self.Q_training.estimate_earning(x, x2)

            r = r * scale

            y_strike = y[:, 2, :]
            r_strike = self.lowest_utility * scale
            loss_strike = loss_gaussion(r_strike, y_strike[:, 0], y_strike[:, 1])
            loss_strike = torch.mean(loss_strike)

            y = y[mask != 2]
            r = r[mask != 2]
            mask = mask[mask != 2]
            y = y[torch.arange(y.shape[0]), mask, :]
            loss = loss_gaussion(r, y[:, 0], y[:, 1])
            weight_class = torch.ones([2]).to(self.device)
            for j in range(2):
                weight_class[j] += torch.sum((mask == j).float())
            weight_class = 1 / weight_class
            weight = weight_class[mask]
            weight = weight * batch_size
            loss = loss * weight
            loss = torch.mean(loss)

            loss += loss_strike

            self.optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Q_training.parameters(), self.max_norm)
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
            loss_list.append(loss.item())

        # self.schedule.step()
        torch.set_grad_enabled(False)
        return np.mean(loss_list)


def calculate_advantage(td_delta, delta_t, worker_id, gamma=0.99, lamada=0.95):
    worker_num = torch.max(worker_id) + 1
    advantage_worker = torch.zeros([int(worker_num)]).to(td_delta.device)
    advantage_list = torch.zeros_like(td_delta).to(td_delta.device)

    for i in range(len(td_delta) - 1, -1, -1):
        advantage_list[i] = advantage_worker[worker_id[i]] * (gamma * lamada) ** delta_t[i] + td_delta[i]
        advantage_worker[worker_id[i]] = advantage_list[i]

    return advantage_list


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


def single_update(observe_space, current_orders, current_orders_num, positive_history, negative_history, speed,
                  capacity, current_travel_route, current_travel_time, experience, feedback, new_route, new_route_time,
                  new_remaining_time, new_total_travel_time, worker_feed_back, reservation_value, time):
    full_experience = None
    finished_order_time = None
    current_orders_num = int(current_orders_num)
    worker_reward = 0
    update_rate = 0.05
    # take action
    if feedback is not None:
        worker_reward = worker_feed_back[1]
        if feedback[-1] == -1:  # -1 means reject
            negative_history = negative_history * (1 - update_rate) + feedback[1][0] * update_rate
        else:  # accept order
            positive_history = positive_history * (1 - update_rate) + feedback[1][0] * update_rate

        # update experience
        if len(experience) > 0:
            experience.append(feedback[0][-1] - experience[0][-1])  # △t
            experience.append([feedback[0], speed, capacity, positive_history, negative_history, time])  # s_next
            experience.append([worker_feed_back, reservation_value])  # worker_next
            # print([speed,capacity,positive_history,negative_history])
            if len(experience) == 7:
                full_experience = experience
            else:
                print("There is a bug (experience)!!")
            experience = []
        experience.append([feedback[0], speed, capacity, positive_history, negative_history, time])  # s_current
        experience.append([worker_feed_back, reservation_value])  # worker_current
        experience.append(feedback[1])  # a
        experience.append(feedback[2])  # r

        # update state
        if feedback[-1] != -1:  # accept order

            observe_space[2] -= 1  # remaining seat
            observe_space[3] = 1  # update to picking up state
            observe_space[4] = feedback[-1]  # remaining picking up time
            observe_space[0] = feedback[0][-2][0]  # plat
            observe_space[1] = feedback[0][-2][1]  # plon

            current_travel_route, current_travel_time = new_route, new_route_time
            current_orders[:current_orders_num + 1, 2], current_orders[:current_orders_num + 1,
                                                        3] = new_remaining_time, new_total_travel_time
            current_orders[current_orders_num, 0], current_orders[current_orders_num, 1] = feedback[0][-2][2], \
            feedback[0][-2][3]  # dlat,dlon (new orders)
            current_orders[current_orders_num, 4] = feedback[1][0]
            current_orders_num += 1

    step = speed * 1  # 1min
    if observe_space[3] == 1:  # pick up
        if observe_space[4] > step:
            observe_space[4] -= step
            step = 0
        else:  # finish picking up
            step -= observe_space[4]
            observe_space[4] = 0
            if observe_space[2] == 0:  # no available seat
                observe_space[3] = 2
            else:  # have available seat
                observe_space[3] = 0
    if step > 0 and current_orders_num != 0:
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
            if i == len(current_travel_time) - 1:  # finish all orders
                observe_space[0], observe_space[1] = current_travel_route[-1][1], current_travel_route[-1][
                    0]  # lat, lon
                current_travel_time = []
                current_travel_route = []

        # print(current_travel_route)
        if len(current_travel_route) > 0:
            observe_space[0], observe_space[1] = current_travel_route[0][1], current_travel_route[0][0]  # lat, lon

        current_orders[:current_orders_num, 2] -= step_minute  # update remaining time

        # delete finished orders
        drop_index = np.zeros(current_orders.shape[0])
        drop_index[:current_orders_num] = (current_orders[:current_orders_num, 2] <= 0)
        drop_num = np.sum(drop_index)
        if drop_num > 0:
            current_orders_num -= drop_num
            observe_space[2] += drop_num
            if observe_space[3] == 2:
                observe_space[3] = 0
            drop_index = drop_index.astype(bool)
            finished_orders = current_orders[drop_index]
            current_orders = current_orders[~drop_index]
            fill_matrix = np.zeros_like(finished_orders)
            current_orders = np.concatenate([current_orders, fill_matrix], axis=0)
            finished_order_time = finished_orders[:, 3] / speed

    return observe_space, current_orders, current_orders_num, positive_history, negative_history, current_travel_route, current_travel_time, experience, full_experience, finished_order_time, worker_reward


def gaussian_entropy(sigma):  # negative entropy
    entropy = 0.5 * torch.log(2 * torch.pi * (sigma ** 2)) + 0.5
    entropy *= -1
    return torch.mean(entropy)

def kl_divergence(mu1, sigma1, mu2, sigma2):
    kl_div1 = torch.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5
    kl_div2 = torch.log(sigma1 / sigma2) + (sigma2 ** 2 + (mu2 - mu1) ** 2) / (2 * sigma1 ** 2) - 0.5
    kl_div = (kl_div1 + kl_div2) / 2
    return torch.mean(kl_div)

def loss_gaussion(y, mu, sigma):
    loss = torch.log(sigma) + (y - mu) ** 2 / (2 * sigma ** 2)
    return torch.mean(loss)

if __name__ == '__main__':
    # test
    # worker=Worker(1000)
    # reservation_value = np.linspace(0.85, 1.15, 1000)
    # worker.reset(reservation_value=reservation_value)
    # import matplotlib.pyplot as plt
    # plt.plot(range(len(worker.positive_history)),worker.positive_history,'r')
    # plt.plot(range(len(worker.positive_history)),worker.negative_history,'b')
    # plt.show()
    plot_accept_rate()
