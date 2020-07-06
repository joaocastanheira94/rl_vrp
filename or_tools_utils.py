import os
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2,pywrapcp
from scipy.spatial.distance import pdist, squareform

from datetime import datetime
from time import sleep
num_vehicles = 20

def preprocess_ortools(capacity,random_data):
    #get depot
    depot = np.array(random_data['depot'])
    #get nodes
    nodes = np.array(random_data['loc'])
    #add depot to index 0
    nodes = np.insert(nodes, 0, depot, 0)

    vehicle_capacities = [capacity for i in range(num_vehicles)]
    
    demands = np.array(random_data['demand'])
    #rescale demands
    demands = demands*capacity
    #insert 0 demand in the depot index
    demands = np.insert(demands,0,0,0).astype(int)
    return depot, nodes, demands, vehicle_capacities, num_vehicles

class cvrp_ortools_solver:
    
    distance_scale = 10**7

    def __init__(self, nodes, num_vehicles, demands, vehicle_capacities):
        self.data = {}
        self.data['distance_matrix'] = self.euclidean_distance_matrix(nodes) * self.distance_scale
        self.data['demands'] = demands
        self.data['vehicle_capacities'] = vehicle_capacities
        self.data['num_vehicles'] = num_vehicles
        self.data['depot'] = 0
        self.manager = None
        self.routing = None

    def euclidean_distance_matrix(self, nodes):
        return squareform(pdist(nodes,metric='euclidean'))

    def evaluate_solution(self, solution,callback):
        df_vrp_solution = pd.DataFrame()
        for vehicle_id in range(self.data['num_vehicles']):
            index,indexes = self.routing.Start(vehicle_id),[]
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            
            route_distance = 0
            route_load = 0
            route_demands = []
            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                indexes.append(node)
                node_demand = self.data['demands'][node]
                route_demands.append(node_demand)
                route_load += node_demand
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                d = self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id) / self.distance_scale
                route_distance += d
                
            r = {'cum_demand':route_load, 'vehicle_id':vehicle_id,'route': indexes[1:],'demands':route_demands[1:],'distance':route_distance,'num_nodes':len(indexes)}
            df_vrp_solution = df_vrp_solution.append(r,ignore_index = True)
        df_vrp_solution = df_vrp_solution[df_vrp_solution['distance']>0.0]

        return df_vrp_solution

    # Create and register a transit callback.
    def distance_callback(self,from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data['distance_matrix'][from_node][to_node]

    #demand callback
    def demand_callback(self, from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        return self.data['demands'][from_node]

    def run(self):

        # Create the routing index manager.
        self.manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']), self.data['num_vehicles'], self.data['depot'])

        # Create Routing Model.
        self.routing = pywrapcp.RoutingModel(self.manager)
            

        callback = self.distance_callback
        transit_callback_index = self.routing.RegisterTransitCallback(callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self.demand_callback)

        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        start = datetime.now()
        # Solve the problem.
        solution = self.routing.SolveWithParameters(search_parameters)
        end = datetime.now()

        if solution:  
            return self.evaluate_solution(solution,callback), (end-start).total_seconds()
        else: 
            return False