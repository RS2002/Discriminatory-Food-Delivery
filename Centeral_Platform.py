import torch
from osrm import TSP_route
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
from Worker import accept_rate, norm

class Platform():
    def __init__(self,discount_factor=0.99):
        super().__init__()
        self.reset(discount_factor)

    def reset(self,discount_factor=0.99):
        self.discount_factor = discount_factor
        self.Total_Reward = 0
        self.Total_Detour = []
        self.PickUp = 0
        self.worker_reject = 0

    def assign(self,q_matrix):
        # TODO: set non-zero threshold
        threshold = 0
        # Solve Bipartite Match Process with ILP
        num_vehicles, num_demands = q_matrix.shape
        Value_Matrix = np.concatenate((q_matrix,np.zeros_like(q_matrix)+threshold),axis=1)
        cost_matrix = -Value_Matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 创建一个列表来保存每个车辆被分配的订单
        assignment = [None] * len(Value_Matrix)

        # 获取每个车辆被分配的订单
        for i in range(len(row_indices)):
            if col_indices[i] >= num_demands:
                assignment[row_indices[i]] = None
            else:
                assignment[row_indices[i]] = col_indices[i]

        # 计算最大化的值
        max_value = -1 * cost_matrix[row_indices, col_indices].sum()

        # 返回分配结果和最大值
        return assignment, max_value

    def feedback(self, observe, reservation_value, speed, current_order, current_order_num, assignment, new_orders_state, price_mu, price_sigma, reward_func, punish_rate, threshold, current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state):
        feedback_table = []
        new_route_table = []
        new_route_time_table = []
        new_remaining_time_table = []
        new_total_travel_time_table = []
        accepted_orders = []
        worker_feed_back_table = []

        results = Parallel(n_jobs=24)(
            delayed(excute)(observe[i], reservation_value[i], speed[i], current_order[i], current_order_num[i], assignment[i], new_orders_state, price_mu[i], price_sigma[i], reward_func, punish_rate, threshold, current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state[i])
            for i
            in range(observe.shape[0]))

        # for i in range(observe.shape[0]):
        #     excute(observe[i], reservation_value[i], current_order[i], current_order_num[i], assignment[i], new_orders_state, price_mu[i], price_sigma[i], reward_func, punish_rate, threshold, current_time)
        # exit()

        for i in range(len(results)):
            result = results[i]
            feedback_table.append(result[0])
            new_route_table.append(result[1])
            new_route_time_table.append(result[2])
            new_remaining_time_table.append(result[3])
            new_total_travel_time_table.append(result[4])
            timeout = result[5]
            if assignment[i] is not None and result[0] is not None:
                self.Total_Reward += self.discount_factor ** current_time * result[0][2] # '''result[0][2]是reward'''
                self.Total_Detour.append(timeout)

            if result[6] is not None:
                accepted_orders.append(result[6])
                self.PickUp += 1
            elif result[0] is not None:
                self.worker_reject += 1
            worker_feed_back_table.append(result[7])

        return feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, accepted_orders, worker_feed_back_table

'''
beta_list:
beta_list[0]: reward of taking new order
beta_list[1]: reward of client paying
beta_list[2]: punishment of paying salary
beta_list[3]: punishment of timeout orders
beta_list[4]: punishment of added time
beta_list[5]: add punishment to the over time
'''
def reward_func_generator(beta_list, threshold):
    def reward(time_add,time_out,salary,direct_distance):
        if time_add <= threshold:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * salary / 100 - beta_list[3] * time_out - beta_list[4] * time_add / 60
        else:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * salary / 100 - beta_list[3] * time_out - beta_list[4] * time_add / 60 - beta_list[5] * (time_add - threshold)
        return r
    return reward


'''
excute for each worker

input:
observe, reservation_value, current_order, current_order_num: worker information
assignment: assignment which order to this worker
new_orders_state: the state matrix of all new orders
price_mu_vector, price_sigma_vector: the price distribution of each order (of this worker)
reward_func, punish_rate, threshold: to calculate the reward

output:
feedback: [[observe, current_order, current_order_num, new_orders_state[assignment], time], [price, price_log_prop], reward, pickup_time] -- (s,a,r,pick_t)
new_route, new_route_time: route/time of each nodes which the worker will pass
new_time, new_total_travel_time: remaining/total time of each order
timeout: how many orders will be timeout after this assignment
'''
def excute(observe, reservation_value, speed, current_order, current_order_num, assignment, new_orders_state, price_mu_vector, price_sigma_vector, reward_func, punish_rate, threshold, current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state):
    current_order_num = int(current_order_num)
    if assignment is not None and observe[3] == 0:
        price_mu, price_sigma = price_mu_vector[assignment], price_sigma_vector[assignment]
        price_dist = torch.distributions.Normal(price_mu, price_sigma)
        price = price_dist.sample()
        price_log_prop = price_dist.log_prob(price)
        price, price_log_prop = price.item(), price_log_prop.item() # avoid gradient propagation

        rand = random.random()
        if rand<exploration_rate:
            acc_rate = 0.5
        else:
            worker_Q_Net.eval()
            torch.set_grad_enabled(False)
            order_norm, x_norm, x_order_norm = norm(np.expand_dims(new_orders_state[assignment], axis=0),
                                                    np.expand_dims(worker_state, axis=0),
                                                    np.expand_dims(current_order, axis=0))
            x = np.concatenate([order_norm, x_norm, np.array([[reservation_value, price]])], axis=-1)
            x = torch.from_numpy(x).to(device)
            x_order_norm = torch.from_numpy(x_order_norm).to(device)
            worker_q_value = worker_Q_Net(x, x_order_norm, torch.tensor([current_order_num]).to(device))
            acc_rate = int(worker_q_value[0, 1]>=worker_q_value[0, 0])

        # worker_Q_Net.eval() # no ε-greedy to avoid platform network cannot converge
        # torch.set_grad_enabled(False)
        # order_norm, x_norm, x_order_norm = norm(np.expand_dims(new_orders_state[assignment], axis=0), np.expand_dims(worker_state, axis=0), np.expand_dims(current_order, axis=0))
        # x = np.concatenate([order_norm, x_norm, np.array([[reservation_value, price]])], axis=-1)
        # x = torch.from_numpy(x).to(device)
        # x_order_norm = torch.from_numpy(x_order_norm).to(device)
        # worker_q_value = worker_Q_Net(x, x_order_norm, torch.tensor([current_order_num]).to(device))
        # acc_rate = worker_q_value[0,1]

        # acc_rate = accept_rate(price,reservation_value)
        rand = random.random()
        if rand<=acc_rate: #accept
            accept_order = assignment
            # 1. direct_distance
            plat, plon, dlat, dlon = new_orders_state[assignment,:4]
            direct_route, direct_route_time, direct_time, direct_distance = TSP_route((plat, plon), [(dlat, dlon)])
            # direct_time = direct_time[0]
            # 2. pickup
            pickup_route, pickup_route_t, pickup_time, _ = TSP_route((observe[0].item(),observe[1].item()), [(plat, plon)])
            # 3. add the new order
            destination_points = []
            for i in range(current_order_num):
                destination_points.append((current_order[i,0].item(),current_order[i,1].item()))
            destination_points.append((dlat, dlon))
            new_route, new_route_time, new_time, _ = TSP_route((plat, plon), destination_points)
            if len(new_route)!=0:
                # 4. calculate reward

                original_total_travel_time = np.sum(current_order[:,3])

                new_total_travel_time = np.array(new_time)
                new_total_travel_time[:-1] = new_total_travel_time[:-1] + current_order[:current_order_num,3] - current_order[:current_order_num,2] # add the time already cost for each old order
                new_total_travel_time = new_total_travel_time + pickup_time[0]

                timeout = np.sum(new_total_travel_time / speed > threshold) # how many orders will be over time
                time_add = np.sum(new_total_travel_time) - original_total_travel_time # total added time of all orders
                # added workload
                if len(new_total_travel_time)>1:
                    work_add = new_total_travel_time[-1] + np.max(new_total_travel_time[:-1] - current_order[:current_order_num,3])
                else:
                    work_add = new_total_travel_time[-1] # no previous work
                salary = work_add * price
                reward = reward_func(time_add / speed, timeout, salary, direct_distance)
                feedback = [[observe,current_order,current_order_num,new_orders_state[assignment], current_time], [price,price_log_prop], reward, pickup_time[0]]

                worker_reward = work_add * (price-reservation_value)
                worker_feedback = [1,worker_reward,price]

                return feedback, new_route, new_route_time, new_time, new_total_travel_time, timeout, accept_order, worker_feedback
            else: # routing failure --> decline the assignment
                return None, None, None, None, None, 0, None, None
        else: #reject
            reward = - punish_rate / price # to help model increase the price
            feedback = [[observe, current_order, current_order_num, new_orders_state[assignment], current_time],
                        [price, price_log_prop], reward, -1]

            worker_feedback = [0,-worker_reject_punishment,price]
            return feedback, None, None, None, None, 0, None, worker_feedback
    else:
        return None, None, None, None, None, 0, None, None