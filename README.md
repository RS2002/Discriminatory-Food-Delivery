# PDF: PPO+DQN Food-delivery

**Article:** "Double PDF: A Multi-action and Multi-agent Reinforcement Learning Framework for Order Dispatch and Salary Decision in Food Delivery" (under way)

**Acknowledgement:** Some parts of the code is based on the work of [‪Yulong Hu‬‬](https://scholar.google.com/citations?user=IfVrhp0AAAAJ&hl=zh-CN&oi=ao).



## 1. Workflow

![](./img/main.png)



![](./img/network.png)



![](./img/workflow.png)





## 2. Dataset 

Due to the copyright, we cannot provide the data used in this paper. But we provide a brief introduction of the format of our data, so that you can use our code in your own dataset.





## 3. How to Run

### 3.0 Prepare

To run our code your own dataset, you need to do some extra setting:

1. Run the docker of [OSRM](https://github.com/Project-OSRM/osrm-backend) (Remember download the corresponding pbf file of your city.)
2. Change the hyper-parameters `norm` function in `Worker.py` (we provide the meaning of them in annotation). You can change them to a proper value in your own environment. For simplify, you can also make the function directly return the input without any processing, even though it may effect model performance.



### 3.1 Pretrain

```shell
python main.py
```



### 3.2 Finetune

```shell
python main.py
```



### 3.3 Eval

```shell
python eval.py
```



## 4. Citation

```

```

