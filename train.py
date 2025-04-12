from Worker import Buffer, Worker, Buffer_Mask
from Centeral_Platform import Platform, reward_func_generator
from Order_Env import Demand
import argparse
import tqdm
import torch
import numpy as np
import pickle
import random

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mask_rate', type=float, default=0)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--train_times', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_clip', type=float, default=0.1)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--converge_epoch', type=int, default=0)
    parser.add_argument('--minimum_episode', type=int, default=500)
    parser.add_argument('--worker_num', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=100000)

    parser.add_argument('--demand_sample_rate', type=float, default=0.6)
    parser.add_argument("--rand_sample_rate", action="store_true",default=False)

    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[3.0,5.0,12.0,3.0,1.0,5.0,0.0])
    parser.add_argument('--reject_punishment', type=float, default=2.0) # punishment for waiting confirmation

    parser.add_argument("--bilstm", action="store_true",default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument("--worker_mode",type=str,default="benchmark") # gdpr, benchmark, mask

    parser.add_argument("--simultaneity_train", action="store_true",default=False)
    parser.add_argument('--lamada', type=float, default=0.9)
    parser.add_argument('--kl_threshold', type=float, default=0.05)

    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--critic_episode', type=int, default=3)
    parser.add_argument('--actor_episode', type=int, default=1)

    parser.add_argument('--epsilon', type=float, default=0.99)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')

    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=12)

    parser.add_argument("--model_path",type=str,default=None)

    parser.add_argument("--intelligent_worker", action="store_true",default=False)
    parser.add_argument('--worker_reject_punishment', type=float, default=0.0)
    parser.add_argument("--probability_worker", action="store_true",default=False)

    parser.add_argument("--demand_path",type=str,default="../data/demand_evening_onehour.csv")
    parser.add_argument("--zone_table_path",type=str,default="../data/zone_table.csv")

    args = parser.parse_args()
    return args

def group_generation_func(worker_num, mode = 2):
    match mode:
        case 1:
            return group_generation_func1(worker_num)
        case 2:
            return group_generation_func2(worker_num)

def group_generation_func1(worker_num):
    reservation_value = np.random.uniform(0.85, 1.15, worker_num)
    speed = np.random.uniform(0.85, 1.15, worker_num)
    capacity = np.random.randint(2, 5, size=1000) # 2,3,4
    group = None
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
    intelligent_worker = args.intelligent_worker
    probability_worker = args.probability_worker

    platform = Platform(discount_factor = args.gamma, njobs = args.njobs, probability_worker = probability_worker)
    demand = Demand(demand_path = args.demand_path)

    buffer = Buffer(capacity = args.buffer_capacity)
    buffer_price = Buffer(capacity = args.buffer_capacity)
    buffer_mask = Buffer_Mask(capacity = args.buffer_capacity)

    worker = Worker(buffer=buffer, buffer_price=buffer_price, buffer_mask=buffer_mask, lr=args.lr, gamma=args.gamma, eps_clip=args.eps_clip, max_step=args.max_step,
                    num=args.worker_num, device=device, zone_table_path=args.zone_table_path, model_path=args.model_path, njobs=args.njobs,
                    intelligent_worker=intelligent_worker, probability_worker = probability_worker, bilstm = args.bilstm, dropout = args.dropout, pretrain = True, worker_mode=args.worker_mode)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)

    worker.Q_training.q_detach = False
    worker.Q_training.p_detach = False

    if intelligent_worker:
        Worker_Q_training = worker.Worker_Q_training
    else:
        Worker_Q_training = None

    best_reward = -1e-8
    best_epoch = 0

    best_epoch_worker = 0

    dic_list = []

    j = args.init_episode
    exploration_rate = max(exploration_rate * (epsilon_decay_rate**j), epsilon_final)

    critic_episode = args.critic_episode
    current_critic_episode = 0
    actor_episode = args.actor_episode
    current_actor_episode = 0

    critic_phase = True

    while True:
        j += 1

        if j==100:
            worker.Q_training.q_detach = True
            worker.Q_training.p_detach = True

        if args.rand_sample_rate:
            demand_sample_rate = random.uniform(0.1, 0.7)
        else:
            demand_sample_rate = args.demand_sample_rate

        c_loss, a_loss, w_loss = 0, 0, 0
        reservation_value, speed, capacity, group = group_generation_func(args.worker_num, args.mode)
        worker.reset(max_step=args.max_step, num=args.worker_num, reservation_value=reservation_value,
                     speed=speed,
                     capacity=capacity, group=group, train=True, demand_sample_rate=demand_sample_rate, mask_rate=args.mask_rate)
        platform.reset(discount_factor=args.gamma)


        demand.reset(episode_time=0, p_sample=demand_sample_rate, wait_time=args.order_max_wait_time)

        if critic_phase:  # train critic
            if current_critic_episode < critic_episode:
                current_critic_episode += 1
            if current_critic_episode == critic_episode:
                current_critic_episode = 0
                critic_phase = False

            exploration_rate = max(exploration_rate * epsilon_decay_rate, epsilon_final)
            exploration_rate_temp = exploration_rate
            print("Exploration Rate: ", exploration_rate_temp)
            worker.buffer = buffer

            mask_rate = torch.sum((worker.mask==1).float()).item() / args.worker_num
            strike_rate = torch.sum((worker.mask==2).float()).item() / args.worker_num
            print("Mask Rate {:}, Strike Rate {:}".format(mask_rate,strike_rate))

            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
                q_value, price_mu, price_sigma, order_state, worker_state = worker.observe(demand.current_demand, t,
                                                                                           exploration_rate_temp)
                assignment, _ = platform.assign(q_value)
                feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, accepted_orders, worker_feed_back_table = platform.feedback(
                    worker.observe_space, worker.reservation_value, worker.speed, worker.current_orders,
                    worker.current_order_num,
                    assignment, order_state, price_mu, price_sigma, reward_func, args.reject_punishment,
                    args.order_threshold, t,
                    Worker_Q_training, exploration_rate_temp, args.worker_reject_punishment, device, worker_state, worker.mask
                )
                worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
                              new_total_travel_time_table, worker_feed_back_table, t, (t == args.max_step - 1), j)
                demand.pickup(accepted_orders)
                demand.update()

                if (t+1)%4==0 and buffer.num > args.batch_size * 2:
                    c_loss, a_loss = worker.train_critic(args.batch_size, 1, show_pbar = False)
            worker.schedule.step()

        else:  # train actor

            # buffer.reset()
            buffer_price.reset()
            if current_actor_episode < actor_episode:
                current_actor_episode += 1
            if current_actor_episode == actor_episode:
                critic_phase = True
                current_actor_episode = 0
            exploration_rate_temp = 0
            print("Exploration Rate: ", exploration_rate_temp)
            worker.buffer = buffer_price

            mask_rate = torch.sum((worker.mask==1).float()).item() / args.worker_num
            strike_rate = torch.sum((worker.mask==2).float()).item() / args.worker_num
            print("Mask Rate {:}, Strike Rate {:}".format(mask_rate,strike_rate))

            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
                q_value, price_mu, price_sigma, order_state, worker_state = worker.observe(demand.current_demand, t,
                                                                                           exploration_rate_temp)
                assignment, _ = platform.assign(q_value)
                feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, accepted_orders, worker_feed_back_table = platform.feedback(
                    worker.observe_space, worker.reservation_value, worker.speed, worker.current_orders,
                    worker.current_order_num,
                    assignment, order_state, price_mu, price_sigma, reward_func, args.reject_punishment,
                    args.order_threshold, t,
                    Worker_Q_training, exploration_rate_temp, args.worker_reject_punishment, device, worker_state, worker.mask
                )
                worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
                              new_total_travel_time_table, worker_feed_back_table, t, (t == args.max_step - 1), j)
                demand.pickup(accepted_orders)
                demand.update()

            c_loss, a_loss = worker.train_actor(j, args.batch_size, args.train_times, update_critic=args.simultaneity_train,lamada=args.lamada,kl_threshold=args.kl_threshold)
            # buffer.reset()
            buffer_price.reset()
            worker.schedule2.step()

        total_pickup = platform.PickUp
        total_reward = platform.Total_Reward / args.worker_num
        average_travel_time = np.mean(worker.Pass_Travel_Time)
        total_timeout = np.sum((np.array(worker.Pass_Travel_Time)>args.order_threshold))
        worker_reject = platform.worker_reject
        worker_reward = np.mean(worker.worker_reward)
        # average_detour = np.mean(np.array(platform.workload) - np.array(platform.direct_time))
        average_detour = average_travel_time - np.mean(platform.direct_time)
        total_valid_distance = np.sum(platform.valid_distance)

        log = "Train Episode {:} , Platform Reward {:} , Worker Reward {:} , Order Pickup {:} , Worker Reject Num {:} , Average Detour {:} , Average Travel Time {:} , Total Timeout Order {:} , Total Valid Distance {:} , Critic Loss {:} , Actor Loss {:} , Worker Loss {:} , Mask Rate {:} , Strike Rate {:}".format(
            j, total_reward, worker_reward, total_pickup, worker_reject, average_detour, average_travel_time, total_timeout, total_valid_distance, c_loss, a_loss, w_loss, mask_rate, strike_rate)
        print(log)
        with open("train.txt", 'a') as file:
            file.write(log+"\n")
        worker.save("latest.pth")

        price_pos = np.mean(platform.price_pos)
        price_neg = np.mean(platform.price_neg)
        price_total = np.mean(np.concatenate((platform.price_pos, platform.price_neg)))
        price_sigma_pos = np.mean(platform.price_sigma_pos)
        price_sigma_neg = np.mean(platform.price_sigma_neg)
        price_sigma_total = np.mean(np.concatenate((platform.price_sigma_pos, platform.price_sigma_neg)))
        price_mu_pos = np.mean(platform.price_mu_pos)
        price_mu_neg = np.mean(platform.price_mu_neg)
        price_mu_total = np.mean(np.concatenate((platform.price_mu_pos, platform.price_mu_neg)))
        print("Price Distribution Mu: Pos {:} , Neg {:}, Total {:}".format(price_mu_pos, price_mu_neg, price_mu_total))
        print("Price Distribution Std: Pos {:} , Neg {:}, Total {:}".format(price_sigma_pos, price_sigma_neg, price_sigma_total))
        print("Real Price Avg: Pos {:} , Neg {:}, Total {:}".format(price_pos, price_neg, price_total))
        print()

        if j % args.eval_episode == 0:
            # mask_phase = not mask_phase
            # worker.buffer_mask.reset()
            # worker.buffer_q.reset()
            # worker.buffer_price.reset()

            worker.schedule.step()
            # worker.schedule2.step()

            reservation_value, speed, capacity, group = group_generation_func(args.worker_num, args.mode)
            worker.reset(max_step=args.max_step, num=args.worker_num, reservation_value=reservation_value, speed=speed,
                         capacity=capacity, group=group, train=False, demand_sample_rate=args.demand_sample_rate, mask_rate=args.mask_rate)
            platform.reset(discount_factor=args.gamma)
            demand.reset(episode_time=0, p_sample=args.demand_sample_rate, wait_time=args.order_max_wait_time)

            print("Eval")
            mask_rate = torch.sum((worker.mask==1).float()).item() / args.worker_num
            strike_rate = torch.sum((worker.mask==2).float()).item() / args.worker_num
            print("Mask Rate {:}, Strike Rate {:}".format(mask_rate,strike_rate))

            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
                q_value, price_mu, price_sigma, order_state, worker_state = worker.observe(demand.current_demand, t,
                                                                                           0)
                assignment, _ = platform.assign(q_value)
                feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, accepted_orders, worker_feed_back_table = platform.feedback(
                    worker.observe_space, worker.reservation_value, worker.speed, worker.current_orders, worker.current_order_num,
                    assignment, order_state, price_mu, price_sigma, reward_func, args.reject_punishment,
                    args.order_threshold, t,
                    Worker_Q_training, 0, args.worker_reject_punishment, device, worker_state, worker.mask
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
            # average_detour = np.mean(np.array(platform.workload) - np.array(platform.direct_time))
            average_detour = average_travel_time - np.mean(platform.direct_time)
            total_valid_distance = np.sum(platform.valid_distance)

            log = "Eval Episode {:} , Platform Reward {:} , Worker Reward {:} , Order Pickup {:} , Worker Reject Num {:} , Average Detour {:} , Average Travel Time {:} , Total Timeout Order {:} , Total Valid Distance {:}, Mask Rate {:} , Strike Rate {:}".format(
                j, total_reward, worker_reward, total_pickup, worker_reject, average_detour, average_travel_time, total_timeout, total_valid_distance, mask_rate, strike_rate
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
            print()

            dic = {
                'episode': j,
                'reservation_value': reservation_value,
                'speed': speed,
                'capacity': capacity,
                'worker_reward': worker.worker_reward,
                'price': worker.price,
                'work_load': worker.work_load,
                'assigned_order': worker.worker_assign_order,
                'reject_order': worker.worker_reject_order,
                'salary': worker.salary,
                'pos_history': worker.positive_history,
                'neg_history': worker.negative_history,
                'price_sigma_pos': platform.price_sigma_pos,
                'price_sigma_neg': platform.price_sigma_neg,
                'mask': worker.mask
            }

            dic_list.append(dic)
            with open('log.pkl', 'wb') as f:
                pickle.dump(dic_list, f)

            if total_reward > best_reward:
                best_epoch = 0
                best_reward = total_reward
                worker.save("best.pth")
            else:
                best_epoch += 1

            if j == args.minimum_episode:
                best_epoch = 0
                best_epoch_worker = 0
            if j >= args.minimum_episode:
                print("Converge Step: ", best_epoch,best_epoch_worker)
                if best_epoch >= args.converge_epoch:
                    break

if __name__ == '__main__':
    main()