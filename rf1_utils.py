import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import torch
from problems import CVRP
from utils import load_model
from datetime import datetime
from time import sleep

class rf1:
    def __init__(self,model_size, data, problem_capacity):
        self.data = data
        self.model, _ = load_model('pretrained/cvrp_{}/'.format(model_size))
        self.batch = None
        self.dataset = None
        self.tours = None
        self.problem_capacity = problem_capacity

    def init_dataset(self):
        self.dataset = CVRP.load_dataset(data = self.data)
        ## Need a dataloader to batch instances
        dataloader = DataLoader(self.dataset, batch_size=1000)
        # Make var works for dicts
        self.batch = next(iter(dataloader))
        self.data = self.data[0]

    def evaluate_solution(self):    
        demands = self.data['demand'].cpu().numpy()
        nodes = self.data['loc'].cpu().numpy()

        df_solution = pd.DataFrame()
        
        for i, (d, tour) in enumerate(zip(self.dataset, self.tours)):
            routes = [r[r!=0] for r in np.split(tour.cpu().numpy(), np.where(tour==0)[0]) if (r != 0).any()]
            total_dist = 0
            for veh_number, r in enumerate(routes):
                route_demands = demands[r - 1]*self.problem_capacity
                coords = nodes[r - 1, :]
                x_dep, y_dep = self.data['depot']
                x_prev, y_prev = x_dep, y_dep
                cum_demand, dist = 0,0
                
                for i2, ((x, y), d) in enumerate(zip(coords, route_demands)):
                    dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
                    x_prev, y_prev = x, y
                    cum_demand += d*self.problem_capacity

                dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
                total_dist += dist

                r = {'vehicle_id':veh_number,'num_nodes':len(coords),'distance':float(dist),'route':r,'cum_demand':route_demands.sum(),'demands':[float(r) for r in route_demands]}
                df_solution = df_solution.append(r,ignore_index= True)
                
        return df_solution

    def run(self):
        torch.manual_seed(1234)
        self.init_dataset()
        self.model.eval()
        self.model.set_decode_type('greedy')
        start = datetime.now()
        with torch.no_grad(): length, log_p, self.tours = self.model(self.batch, return_pi=True)
        end = datetime.now()
        return self.evaluate_solution(),(end-start).total_seconds()