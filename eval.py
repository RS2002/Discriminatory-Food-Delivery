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
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--converge_epoch', type=int, default=10)
    parser.add_argument('--minimum_episode', type=int, default=500)
    parser.add_argument('--worker_num', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=30000)
    parser.add_argument('--demand_sample_rate', type=float, default=0.2)
    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[3.0,5.0,4.0,3.0,1.0,5.0,0.0])
    parser.add_argument('--reject_punishment', type=float, default=0.0)

    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')

    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=24)

    parser.add_argument("--platform_model_path",type=str,default="./platform_best.pth")

    parser.add_argument("--intelligent_worker", action="store_true",default=False)
    parser.add_argument('--worker_reject_punishment', type=float, default=0.0)
    parser.add_argument("--worker_model_path",type=str,default=None)
    parser.add_argument("--probability_worker", action="store_true",default=False)


    parser.add_argument("--demand_path",type=str,default="./data/demand_evening_onehour.csv")
    parser.add_argument("--zone_table_path",type=str,default="./data/zone_table.csv")
    parser.add_argument('--eval_times', type=int, default=10)

    args = parser.parse_args()
    return args

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

    intelligent_worker = args.intelligent_worker
    probability_worker = args.probability_worker

    platform = Platform(discount_factor = args.gamma, njobs = args.njobs, probability_worker = probability_worker)
    demand = Demand(demand_path = args.demand_path)
    buffer = Buffer(capacity = args.buffer_capacity)
    worker = Worker(buffer=buffer, lr=args.lr, gamma=args.gamma, eps_clip=args.eps_clip, max_step=args.max_step,
                    num=args.worker_num,
                    device=device, zone_table_path=args.zone_table_path, model_path=args.platform_model_path,
                    worker_model_path=args.worker_model_path, njobs=args.njobs, intelligent_worker=intelligent_worker, probability_worker = probability_worker)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)

    Worker_Q_training = None

    dic_list = []

    for j in range(args.eval_times):
        reservation_value, speed, capacity, group = group_generation_func2(args.worker_num)
        worker.reset(max_step=args.max_step, num=args.worker_num, reservation_value=reservation_value, speed=speed,
                     capacity=capacity, group=group, train=False)
        platform.reset(discount_factor=args.gamma)
        demand.reset(episode_time=0, p_sample=args.demand_sample_rate, wait_time=args.order_max_wait_time)
        pbar = tqdm.tqdm(range(args.max_step))
        for t in pbar:
            q_value, price_mu, price_sigma, order_state, worker_state = worker.observe(demand.current_demand, t,
                                                                                       0)
            assignment, _ = platform.assign(q_value)
            feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, accepted_orders, worker_feed_back_table = platform.feedback(
                worker.observe_space, worker.reservation_value, worker.speed, worker.current_orders,
                worker.current_order_num,
                assignment, order_state, price_mu, price_sigma, reward_func, args.reject_punishment,
                args.order_threshold, t,
                Worker_Q_training, 0, args.worker_reject_punishment, device, worker_state
            )
            worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
                          new_total_travel_time_table, worker_feed_back_table, t, (t == args.max_step - 1), j)
            demand.pickup(accepted_orders)
            demand.update()

        total_pickup = platform.PickUp
        total_reward = platform.Total_Reward / args.worker_num
        average_travel_time = np.mean(worker.Pass_Travel_Time)
        total_timeout = np.sum((np.array(worker.Pass_Travel_Time) > args.order_threshold))
        worker_reject = platform.worker_reject
        worker_reward = np.mean(worker.worker_reward)
        average_detour = np.mean(np.array(platform.workload) - np.array(platform.direct_time))
        total_valid_distance = np.sum(platform.valid_distance)

        log = "Eval Episode {:} , Platform Reward {:} , Worker Reward {:} , Order Pickup {:} , Worker Reject Num {:} , Average Detour {:} , Average Travel Time {:} , Total Timeout Order {:} , Total Valid Distance {:}".format(
            j, total_reward, worker_reward, total_pickup, worker_reject, average_detour, average_travel_time,
            total_timeout, total_valid_distance
        )
        print(log)
        with open("eval.txt", 'a') as file:
            file.write(log + "\n")

        price_pos = np.mean(platform.price_pos)
        price_neg = np.mean(platform.price_neg)
        price_total = np.mean(np.concatenate((platform.price_pos, platform.price_neg)))
        price_sigma_pos = np.mean(platform.price_sigma_pos)
        price_sigma_neg = np.mean(platform.price_sigma_neg)
        price_sigma_total = np.mean(np.concatenate((platform.price_sigma_pos, platform.price_sigma_neg)))
        price_mu_pos = np.mean(platform.price_mu_pos)
        price_mu_neg = np.mean(platform.price_mu_neg)
        price_mu_total = np.mean(np.concatenate((platform.price_mu_pos, platform.price_mu_neg)))
        print("Price Distribution Mu: Pos {:} , Neg {:}, Total {:}".format(price_mu_pos, price_mu_neg,
                                                                           price_mu_total))
        print("Price Distribution Std: Pos {:} , Neg {:}, Total {:}".format(price_sigma_pos, price_sigma_neg,
                                                                            price_sigma_total))
        print("Real Price Avg: Pos {:} , Neg {:}, Total {:}".format(price_pos, price_neg, price_total))

        dic = {
            'episode': j,
            'reservation_value': reservation_value,
            'worker_reward': worker.worker_reward,
            'price': worker.price,
            'work_load': worker.work_load,
            'assigned_order': worker.worker_assign_order,
            'reject_order': worker.worker_reject_order,
            'salary': worker.salary,
            'pos_history': worker.positive_history,
            'neg_history': worker.negative_history,
            'price_sigma_pos': platform.price_sigma_pos,
            'price_sigma_neg': platform.price_sigma_neg
        }

        dic_list.append(dic)
        with open('log_eval.pkl', 'wb') as f:
            pickle.dump(dic_list, f)

if __name__ == '__main__':
    main()