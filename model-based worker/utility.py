#公式计算两点间距离（m）
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import radians, cos, sin, asin, sqrt
def geodistance(lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def zone_order_map(action):
    if action % 57 == 0:
        index = action // 57 - 1
        zone = 56
    else:
        index = action // 57
        zone = (action % 57) - 1

    return index, zone


def time_record(start, end1, end2,end3, end4, end5, end6):
    print("Vehicles observation takes {0} seconds".format(end1 - start))
    print("Vehicles  decision takes {0} seconds".format(end2 - end1))
    print("Central assignment takes {0} seconds".format(end3 - end2))
    print("Central feedback process take {0} seconds".format(end4 - end3))
    print("Vehicles update process take {0} seconds".format(end5 - end4))
    print("Vehicles learning process take {0} seconds".format(end6 - end5))
    print("1 round of whole process take {0} seconds".format(end6 - start))

def training_curve(training_loss_steps, total_reward_eps, contradiction_rate_eps, average_detour_eps,
                   validation_reward_eps, curve_save_path):
    s1 = pd.Series(training_loss_steps)
    rolling_avg1 = s1.rolling(window=1000).mean()
    # rolling_max1 = s1.rolling(window=30).max()
    # rolling_min1 = s1.rolling(window=30).min()
    rolling_avg_list1 = rolling_avg1.tolist()
    # rolling_max_list1 = rolling_max1.tolist()
    # rolling_min_list1 = rolling_min1.tolist()

    s2 = pd.Series(total_reward_eps)
    rolling_avg2 = s2.rolling(window=20).mean()
    rolling_max2 = s2.rolling(window=20).max()
    rolling_min2 = s2.rolling(window=20).min()
    rolling_avg_list2 = rolling_avg2.tolist()
    rolling_max_list2 = rolling_max2.tolist()
    rolling_min_list2 = rolling_min2.tolist()

    s3 = pd.Series(contradiction_rate_eps)
    rolling_avg3 = s3.rolling(window=1).mean()
    # rolling_max3 = s3.rolling(window=10).max()
    # rolling_min3 = s3.rolling(window=10).min()
    rolling_avg_list3 = rolling_avg3.tolist()
    # rolling_max_list3 = rolling_max3.tolist()
    # rolling_min_list3 = rolling_min3.tolist()

    s4 = pd.Series(average_detour_eps)
    rolling_avg4 = s4.rolling(window=20).mean()
    # rolling_max4 = s4.rolling(window=10).max()
    # rolling_min4 = s4.rolling(window=10).min()
    rolling_avg_list4 = rolling_avg4.tolist()
    # rolling_max_list4 = rolling_max4.tolist()
    # rolling_min_list4 = rolling_min4.tolist()

    s5 = pd.Series(validation_reward_eps)
    rolling_avg5 = s5.rolling(window=1).mean()
    # rolling_max4 = s4.rolling(window=10).max()
    # rolling_min4 = s4.rolling(window=10).min()
    rolling_avg_list5 = rolling_avg5.tolist()
    # rolling_max_list4 = rolling_max4.tolist()
    # rolling_min_list4 = rolling_min4.tolist()

    # Create a figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    # Plot the first graph on the first subplot
    ax1.plot(rolling_avg_list1)
    # ax1.plot(rolling_max_list1)
    # ax1.plot(rolling_min_list1)
    # ax1.fill_between(range(len(rolling_min_list1)), rolling_min_list1, rolling_max_list1, alpha=0.5)
    ax1.set_xlabel('training steps')
    ax1.set_ylabel('training loss')
    ax1.set_title('loss')

    # Plot the second graph on the second subplot
    ax2.plot(rolling_avg_list2)
    ax2.plot(rolling_max_list2)
    ax2.plot(rolling_min_list2)
    ax2.fill_between(range(len(rolling_min_list2)), rolling_min_list2, rolling_max_list2, alpha=0.5)
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('total reward')
    ax2.set_title('total reward')

    # Plot the third graph on the second subplot
    ax3.plot(rolling_avg_list3)
    # ax3.plot(rolling_max_list3)
    # ax3.plot(rolling_min_list3)
    # ax3.fill_between(range(len(rolling_min_list3)), rolling_min_list3, rolling_max_list3, alpha=0.5)
    ax3.set_xlabel('episodes')
    ax3.set_ylabel('Estimation reward')
    ax3.set_title('Estimation')

    ax4.plot(rolling_avg_list4)
    # ax4.plot(rolling_max_list4)
    # ax4.plot(rolling_min_list4)
    # ax4.fill_between(range(len(rolling_min_list4)), rolling_min_list4, rolling_max_list4, alpha=0.5)
    ax4.set_xlabel('episodes')
    ax4.set_ylabel('average detour')
    ax4.set_title('Detour')

    ax5.plot(rolling_avg_list5)
    # ax4.plot(rolling_max_list4)
    # ax4.plot(rolling_min_list4)
    # ax4.fill_between(range(len(rolling_min_list4)), rolling_min_list4, rolling_max_list4, alpha=0.5)
    ax5.set_xlabel('episodes')
    ax5.set_ylabel('Rewards')
    ax5.set_title('Validation')

    plt.savefig(curve_save_path)  # Show the plot 'Trainingplot/training_0725.png'
    plt.show()

def training_record(training_loss_steps, total_reward_eps, contradiction_rate_eps, average_detour_eps,
                    validation_reward_eps, record_save_path):
    with open(record_save_path, 'wb') as f:
        # 使用pickle.dump()将列表写入文件
        pickle.dump(training_loss_steps, f)
        pickle.dump(total_reward_eps, f)
        pickle.dump(contradiction_rate_eps, f)
        pickle.dump(average_detour_eps, f)
        pickle.dump(validation_reward_eps, f)