import pandas as pd

class Demand():
    def __init__(self, demand_path):
        self.demand = pd.read_csv(demand_path)
        self.filtered_demand = self.demand.loc[self.demand['minute'] == 0].reset_index(drop=True)
        self.current_demand = self.demand.loc[self.demand['minute'] == 0].reset_index(drop=True)
        self.episode_time = 0
        self.current_time = 0
        self.num_lost_demand = 0

    '''
    episode_time: start minute in this episode
    p_sample: randomly select 100p% samples from the dataset
    wait_time: the maximum waiting time of each order
    '''
    def reset(self, episode_time=0, p_sample=0.95, wait_time=5):
        self.filtered_demand = self.demand.sample(frac=p_sample).sort_index()
        mask = (self.filtered_demand['minute'] >= episode_time) & (self.filtered_demand['minute'] <= 60)
        print("p_sample is:", p_sample)
        print("total number of demand at this episode is:", len(self.filtered_demand[mask]))
        self.current_demand = self.filtered_demand.loc[self.filtered_demand['minute'] == episode_time].reset_index(drop=True)
        self.episode_time = episode_time
        self.current_time = episode_time
        self.num_lost_demand = 0
        self.wait_time = wait_time

    '''
    update the order in the next minute
    throw away orders waiting longer than <wait_time> minutes
    '''
    def update(self):
        self.current_time += 1
        self.current_demand = pd.concat(
            [self.current_demand, self.filtered_demand.loc[self.filtered_demand['minute'] == self.current_time]])
        self.current_demand = self.current_demand.reset_index(drop=True)

        # drop those orders that are not taken over <wait_time> minutes
        if self.current_time >= self.wait_time + self.episode_time:
            self.num_lost_demand += len(self.current_demand[self.current_demand['minute'] <= (self.current_time - self.wait_time)])
            self.current_demand = self.current_demand.drop(
                index=self.current_demand[self.current_demand['minute'] <= (self.current_time - self.wait_time)].index).reset_index(
                drop=True)

    '''
    delete the accepted orders from current_demand list 
    '''
    def pickup(self,unique_r_ids):
        # Convert the set to a list
        unique_r_ids_list = list(unique_r_ids)

        # Drop rows whose index is in unique_r_ids_list
        self.current_demand = self.current_demand.drop(unique_r_ids_list)

        # Reset index
        self.current_demand = self.current_demand.reset_index(drop=True)