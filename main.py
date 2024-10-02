from Worker import Buffer, Worker
from Centeral_Platform import Platform, reward_func_generator
from Order_Env import Demand
import argparse
import tqdm
import torch
import numpy as np
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--train_times', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--max_step', type=int, default=60) #test: 10
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--converge_epoch', type=int, default=5)
    parser.add_argument('--minimum_episode', type=int, default=500)
    parser.add_argument('--worker_num', type=int, default=1000) #test: 50
    parser.add_argument('--buffer_capacity', type=int, default=1e5)
    parser.add_argument('--demand_sample_rate', type=float, default=0.95)
    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[5.0,3.0,3.0,2.0,1.0,4.0]) #ori: 10.0,5.0,3.0,2.0,1.0,4.0
    parser.add_argument('--reject_punishment', type=float, default=0.0)
    parser.add_argument('--worker_reject_punishment', type=float, default=0.0)

    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')

    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=24)

    parser.add_argument("--platform_model_path",type=str,default=None)
    parser.add_argument("--worker_model_path",type=str,default=None)

    parser.add_argument("--demand_path",type=str,default="./data/demand_evening_onehour.csv")
    parser.add_argument("--zone_table_path",type=str,default="./data/zone_table.csv")

    args = parser.parse_args()
    return args


def group_generation_func1(worker_num):
    one_group_worker = worker_num // 2

    group = [0]*one_group_worker
    group.extend([1]*one_group_worker)
    group = np.array(group)
    capacity = np.array([3.0]*worker_num)

    # reservation_value_0 = np.random.normal(loc=0.95, scale=0.01, size=one_group_worker)
    # reservation_value_1 = np.random.normal(loc=0.98, scale=0.01, size=one_group_worker)
    # reservation_value = np.concatenate([reservation_value_0,reservation_value_1])
    # reservation_value[reservation_value<0.9]=0.9

    reservation_value_0 = np.random.normal(loc=0.9, scale=0.05, size=one_group_worker)
    reservation_value_1 = np.random.normal(loc=1.1, scale=0.05, size=one_group_worker)
    reservation_value = np.concatenate([reservation_value_0,reservation_value_1],axis=0)
    reservation_value[reservation_value<0.5]=0.5

    speed_0 = np.random.normal(loc=0.98, scale=0.01, size=one_group_worker)
    speed_1 = np.random.normal(loc=1.00, scale=0.01, size=one_group_worker)
    speed = np.concatenate([speed_0,speed_1],axis=0)
    speed[speed<0.9]=0.9

    return reservation_value, speed, capacity, group

def group_generation_func2(worker_num):
    reservation_value = np.random.uniform(0.85, 1.15, worker_num)
    speed = np.array([1.0]*worker_num)
    capacity = np.array([3.0]*worker_num)
    group = None
    return reservation_value, speed, capacity, group


def main():
    args = get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    exploration_rate = args.epsilon
    epsilon_decay_rate = args.epsilon_decay_rate
    epsilon_final = args.epsilon_final

    platform = Platform(discount_factor = args.gamma, njobs = args.njobs)
    demand = Demand(demand_path = args.demand_path)
    buffer = Buffer(capacity = args.buffer_capacity)
    worker = Worker(buffer = buffer, lr = args.lr, gamma = args.gamma, eps_clip = args.eps_clip, max_step = args.max_step,
                    num = args.worker_num,
                    device = device, zone_table_path=args.zone_table_path, model_path = args.platform_model_path, worker_model_path = args.worker_model_path, njobs = args.njobs)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)


    best_reward = -1e-8
    best_epoch = 0

    j = args.init_episode
    exploration_rate = max(exploration_rate * (epsilon_decay_rate**j), epsilon_final)

    while True:
        j += 1

        reservation_value, speed, capacity, group = group_generation_func2(args.worker_num)
        worker.reset(max_step=args.max_step, num= args.worker_num, reservation_value=reservation_value, speed=speed, capacity=capacity, group=group)
        platform.reset(discount_factor = args.gamma)
        demand.reset(episode_time=0, p_sample=args.demand_sample_rate, wait_time=args.order_max_wait_time)
        exploration_rate = max(exploration_rate * epsilon_decay_rate, epsilon_final)
        print("Exploration Rate: ",exploration_rate)

        pbar = tqdm.tqdm(range(args.max_step))
        for t in pbar:
        # for _ in range(args.max_step):
            q_value, price_mu, price_sigma, order_state, worker_state = worker.observe(demand.current_demand, t, exploration_rate)
            assignment, _ = platform.assign(q_value)
            feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, accepted_orders, worker_feed_back_table = platform.feedback(
                worker.observe_space, worker.reservation_value, worker.speed, worker.current_orders, worker.current_order_num,
                assignment, order_state, price_mu, price_sigma, reward_func, args.reject_punishment, args.order_threshold, t,
                worker.Worker_Q_training, exploration_rate * 0.5, args.worker_reject_punishment, device, worker_state
            )
            worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, worker_feed_back_table, t, (t==args.max_step-1))
            demand.pickup(accepted_orders)
            demand.update()

        c_loss, a_loss, w_loss = worker.train(args.batch_size, args.train_times)

        total_pickup = platform.PickUp
        total_reward = platform.Total_Reward / args.worker_num
        # average_detour = np.mean(platform.Total_Detour)
        average_travel_time = np.mean(worker.Pass_Travel_Time)
        total_timeout = np.sum((np.array(worker.Pass_Travel_Time)>args.order_threshold))
        worker_reject = platform.worker_reject
        worker_reward = np.mean(worker.worker_reward)
        average_detour = np.mean(np.array(platform.workload) - np.array(platform.direct_time))
        total_valid_distance = np.sum(platform.valid_distance)

        log = "Train Episode {:} , Platform Reward {:} , Worker Reward {:} , Order Pickup {:} , Worker Reject Num {:} , Average Detour {:} , Average Travel Time {:} , Total Timeout Order {:} , Total Valid Distance {:} , Critic Loss {:} , Actor Loss {:}, Worker Loss {:}".format(
            j, total_reward, worker_reward, total_pickup, worker_reject, average_detour, average_travel_time, total_timeout, total_valid_distance, c_loss, a_loss, w_loss
        )
        print(log)
        with open("train.txt", 'a') as file:
            file.write(log+"\n")
        worker.save("platform_latest.pth", "worker_latest.pth")

        if j % args.eval_episode == 0:
            reservation_value, speed, capacity, group = group_generation_func2(args.worker_num)
            worker.reset(max_step=args.max_step, num=args.worker_num, reservation_value=reservation_value, speed=speed,
                         capacity=capacity, group=group)
            platform.reset(discount_factor=args.gamma)
            demand.reset(episode_time=0, p_sample=args.demand_sample_rate, wait_time=args.order_max_wait_time)
            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
            # for _ in range(args.max_step):
                q_value, price_mu, price_sigma, order_state, worker_state = worker.observe(demand.current_demand, t,
                                                                                           0)
                assignment, _ = platform.assign(q_value)
                feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, accepted_orders, worker_feed_back_table = platform.feedback(
                    worker.observe_space, worker.reservation_value, worker.speed, worker.current_orders, worker.current_order_num,
                    assignment, order_state, price_mu, price_sigma, reward_func, args.reject_punishment,
                    args.order_threshold, t,
                    worker.Worker_Q_training, 0, args.worker_reject_punishment, device, worker_state
                )
                worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
                              new_total_travel_time_table, worker_feed_back_table, t, (t == args.max_step - 1))
                demand.pickup(accepted_orders)
                demand.update()

            total_pickup = platform.PickUp
            total_reward = platform.Total_Reward / args.worker_num
            # average_detour = np.mean(platform.Total_Detour)
            average_travel_time = np.mean(worker.Pass_Travel_Time)
            total_timeout = np.sum((np.array(worker.Pass_Travel_Time) > args.order_threshold))
            worker_reject = platform.worker_reject
            worker_reward = np.mean(worker.worker_reward)
            average_detour = np.mean(np.array(platform.workload) - np.array(platform.direct_time))
            total_valid_distance = np.sum(platform.valid_distance)

            log = "Eval Episode {:} , Platform Reward {:} , Worker Reward {:} , Order Pickup {:} , Worker Reject Num {:} , Average Detour {:} , Average Travel Time {:} , Total Timeout Order {:} , Total Valid Distance {:}".format(
                j, total_reward, worker_reward, total_pickup, worker_reject, average_detour, average_travel_time, total_timeout, total_valid_distance
            )
            print(log)
            with open("eval.txt", 'a') as file:
                file.write(log + "\n")

            dic = {
                'reservation_value': reservation_value,
                'worker_reward': worker.worker_reward,
                'price': worker.price,
                'work_load': worker.work_load,
                'assigned_order': worker.worker_assign_order,
                'salary': worker.salary,
                'pos_history': worker.positive_history,
                'neg_history': worker.negative_history
            }
            with open('log.pkl', 'wb') as f:
                pickle.dump(dic, f)

            if j >= args.minimum_episode:
                if total_reward + worker_reward * 30 > best_reward:
                    best_epoch = 0
                    best_reward = total_reward + worker_reward * 30
                    worker.save("platform_best.pth", "worker_best.pth")
                else:
                    best_epoch += 1
                print("Converge Step: ", best_epoch)

                if best_epoch >= args.converge_epoch :
                    break


if __name__ == '__main__':
    main()