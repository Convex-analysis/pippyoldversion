import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class Vehicle:
    def __init__(self, gpu_performance, memory, trajectory):
        self.gpu_performance = gpu_performance
        self.memory = memory
        self.trajectory = trajectory

    def get_specs(self):
        return {
            'gpu_performance': self.gpu_performance,
            'memory': self.memory,
            'trajectory': self.trajectory
        }

    def predict_dwell_time(self, historical_data):
        X = np.array([b for b in self.trajectory])
        y = np.array([dwl for dwl in historical_data])
        model = LinearRegression()
        model.fit(X, y)
        return model.predict(X)

    def evaluate_availability(self, historical_data, M_cmp, e_req, M_cap):
        cmp_i = self.gpu_performance
        mem_i = self.memory
        dwl_i = self.predict_dwell_time(historical_data)
        is_sufficient = dwl_i * cmp_i >= M_cmp * e_req and mem_i >= M_cap
        return is_sufficient, cmp_i, mem_i, dwl_i

    @staticmethod
    def from_csv(file_path):
        df = pd.read_csv(file_path)
        vehicles = []
        for _, row in df.iterrows():
            vehicle = Vehicle(row['gpu_performance'], row['memory'], row['trajectory'])
            vehicles.append(vehicle)
        return vehicles

    @staticmethod
    def generate_random(num_vehicles, gpu_range, memory_range, trajectory_length):
        vehicles = []
        for _ in range(num_vehicles):
            gpu_performance = np.random.uniform(*gpu_range)
            memory = np.random.uniform(*memory_range)
            trajectory = np.random.rand(trajectory_length, 2)  # Assuming 2D trajectory
            vehicle = Vehicle(gpu_performance, memory, trajectory)
            vehicles.append(vehicle)
        return vehicles
