# DOUBLE-PDF: DOUBLE Ppo and double Dqn for on-demand Food delivery

**Article:** Zijian Zhao, Sen Li*, "Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework" (under revise)

**Notice: We have identified some minor bugs in the original version of code. You can find the latest code at [DFD](https://github.com/RS2002/DFD).**

**Acknowledgement:** Some parts of the code is based on the work of [‪Yulong Hu‬‬](https://scholar.google.com/citations?user=IfVrhp0AAAAJ&hl=zh-CN&oi=ao).

## 1. Workflow

![](./img/main.png)



![](./img/network.png)



## 2. Dataset 

Due to copyright restrictions, we cannot provide the data used in this paper. However, we offer a brief introduction to the data format so you can utilize our code with your own dataset.

Our dataset consists of one hour of food delivery data in Hong Kong, China, containing approximately 10,000 records. It is saved in a CSV file, where each column represents an attribute and each row corresponds to an order. The relevant attributes include:

1. **dlat**: Latitude of the destination
2. **dlon**: Longitude of the destination
3. **plat**: Latitude of the origin
4. **plon**: Longitude of the origin
5. **minute**: The minute at which the order is placed

As you can see, there is no ground truth for salary information. Therefore, we simply set the reservation value to range from 0.85 to 1.15, without a specific unit.

## 3. Citation

```

```

