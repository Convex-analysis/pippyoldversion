import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from .vehicle import Vehicle

class StaticPlanning:
    def __init__(self, historical_data):
        self.historical_data = historical_data

    def filter_sufficient_vehicles(self, vehicles, M_cmp, e_req, M_cap):
        sufficient_vehicles = []
        for vehicle in vehicles:
            is_sufficient, cmp_i, mem_i, dwl_i = vehicle.evaluate_availability(self.historical_data, M_cmp, e_req, M_cap)
            if is_sufficient:
                sufficient_vehicles.append(vehicle)
        return sufficient_vehicles

    def cluster_vehicles(self, vehicles, M_cmp, e_req, M_cap, alpha):
        clusters = []
        for vehicle in vehicles:
            is_sufficient, cmp_i, mem_i, dwl_i = vehicle.evaluate_availability(self.historical_data, M_cmp, e_req, M_cap)
            if not is_sufficient:
                cluster = self.form_cluster(vehicle, vehicles, M_cmp, e_req, M_cap, alpha)
                clusters.append(cluster)
        return clusters

    def form_cluster(self, vehicle, vehicles, M_cmp, e_req, M_cap, alpha):
        cluster = [vehicle]
        for nb in vehicles:
            if nb != vehicle:
                is_sufficient, cmp_nb, mem_nb, dwl_nb = nb.evaluate_availability(self.historical_data, M_cmp, e_req, M_cap)
                if dwl_nb * cmp_nb >= alpha * M_cmp * e_req and mem_nb >= M_cap:
                    cluster.append(nb)
        return cluster

    def stability_score(self, vehicle, neighbors, dwl_v):
        stability = 0
        for t in range(dwl_v):
            for nb in neighbors:
                stability += self.relative_distance(vehicle, nb, t)
        return stability

    def relative_distance(self, vehicle, neighbor, t):
        C = self.get_cells()
        mobility_patterns = self.get_mobility_patterns()
        P_mobility = self.get_mobility_probabilities(vehicle, neighbor)
        
        P_c_v = self.get_transition_probabilities(vehicle, mobility_patterns)
        P_c_nb = self.get_transition_probabilities(neighbor, mobility_patterns)
        
        P_c_f = np.sum([P_mobility[m] * P_c_v[m] * P_c_nb[m] for m in mobility_patterns], axis=0)
        
        RD_nb = 0
        for c_nb_f in C:
            for c_v_f in C:
                RD_nb += P_c_f[c_nb_f, c_v_f] * self.cell_distance(c_v_f, c_nb_f)
        return RD_nb

    def get_cells(self):
        # Define the cells based on the area and unit distance
        pass

    def get_mobility_patterns(self):
        # Define the mobility patterns
        pass

    def get_mobility_probabilities(self, vehicle, neighbor):
        # Calculate the probability of following each mobility pattern
        pass

    def get_transition_probabilities(self, entity, mobility_patterns):
        # Calculate the transition probabilities for each mobility pattern
        pass

    def cell_distance(self, cell1, cell2):
        # Calculate the distance between two cells
        pass

    def optimize_clusters(self, vehicles, M_cmp, e_req, M_cap, alpha, alpha_prime):
        clusters = self.cluster_vehicles(vehicles, M_cmp, e_req, M_cap, alpha)
        optimized_clusters = []
        for cluster in clusters:
            if self.is_cluster_sufficient(cluster, M_cmp, e_req, M_cap, alpha_prime):
                optimized_clusters.append(cluster)
        return optimized_clusters

    def is_cluster_sufficient(self, cluster, M_cmp, e_req, M_cap, alpha_prime):
        total_mem = sum([v.get_specs()['memory'] for v in cluster])
        total_cmp = sum([v.get_specs()['gpu_performance'] for v in cluster])
        total_dwl = sum([v.predict_dwell_time(self.historical_data) for v in cluster])
        return total_mem > M_cap and total_dwl * total_cmp > e_req * alpha_prime * M_cmp
