"""
Credit: Tianwei Yin
"""

from os import stat
import logging
from torch_scatter import scatter_mean, scatter_max
from torch import nn
from .registry import register_model
import numpy as np
import torch
import pippy

_logger = logging.getLogger(__name__)

class LidarModel(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32],
        backbone='conv',
        min_x=-20, max_x=30,
        min_y=-25, max_y=25,
        pixels_per_meter=4,
        output_features=256):

        super().__init__()

        self.point_pillar_net = PointPillarNet(
            num_input, num_features,
            min_x=min_x, max_x=max_x,
            min_y=min_y, max_y=max_y,
            pixels_per_meter=pixels_per_meter,
        )

        num_feature = num_features[-1]
        self.backbone = ConvBackbone(num_feature=num_feature)
        self.reduce_size = nn.Conv2d(6*num_feature, output_features, 2, 2)

    def forward(self, lidars, num_points):
        features = self.point_pillar_net(lidars, num_points)
        features = self.backbone(features)
        features = self.reduce_size(features)
        features = features[:, :, 5:55, 5:55]
        return features

class DynamicPointNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32]):
        super().__init__()

        L = []
        for num_feature in num_features:
            L += [
                nn.Linear(num_input, num_feature),
                nn.BatchNorm1d(num_feature),
                nn.ReLU(inplace=True),
            ]

            num_input = num_feature

        self.net = nn.Sequential(*L)

    def forward(self, points, inverse_indices):
        """
        TODO: multiple layers
        """
        feat = self.net(points)
        feat_max = scatter_max(feat, inverse_indices, dim=0)[0]
        return feat_max

pippy.fx.wrap('len')

def process_lidar_batch(self, lidar_list, num_points):
        coords = []
        filtered_points = []
        for batch_id, points in enumerate(lidar_list):
            points = points[:num_points[batch_id]]
            points, grid_yx = self.grid_locations(points)

            # batch indices 
            grid_byx = torch.nn.functional.pad(grid_yx, (1, 0), mode='constant', value=batch_id)

            coords.append(grid_byx)
            filtered_points.append(points)

pippy.fx.wrap('process_lidar_batch')

def custom_cat(coords, filtered_points):
        coords = torch.cat(coords, dim=0)
        filtered_points = torch.cat(filtered_points, dim=0)
        return coords, filtered_points
pippy.fx.wrap('custom_cat')
def custom_ones(tensor, device):
    return torch.ones(tensor, device=device)
pippy.fx.wrap('custom_ones')

# Define a custom function
def custom_decorate(points, unique_coords, inverse_indices, pixels_per_meter, min_x, min_y):
    dtype = points.dtype
    x_centers = unique_coords[inverse_indices][:, 2:3].to(dtype) / pixels_per_meter + min_x
    y_centers = unique_coords[inverse_indices][:, 1:2].to(dtype) / pixels_per_meter + min_y

    xyz = points[:, :3]

    # Use scatter_mean to avoid dynamic control flow
    points_cluster = xyz - scatter_mean(xyz, inverse_indices, dim=0)[inverse_indices]

    points_xp = xyz[:, :1] - x_centers
    points_yp = xyz[:, 1:2] - y_centers

    features = torch.cat([points, points_cluster, points_xp, points_yp], dim=-1)
    return features

# Wrap the custom function
pippy.fx.wrap('custom_decorate')

def custom_zero_tensor(batch_size, features, ny, nx, dtype, device):
    return torch.zeros(batch_size, features.shape[1], ny, nx, dtype=features.dtype, device=features.device)
pippy.fx.wrap('custom_zero_tensor')    

def custom_scatter_points(features, coords, batch_size, ny, nx):
    canvas = torch.zeros(batch_size, features.shape[1], ny, nx, dtype=features.dtype, device=features.device)
    canvas[coords[:, 0], :, torch.clamp(ny - 1 - coords[:, 1], 0, ny - 1), torch.clamp(coords[:, 2], 0, nx - 1)] = features
    return canvas
pippy.fx.wrap('custom_scatter_points') 
class PointPillarNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32],
        min_x=-20, max_x=30,
        min_y=-25, max_y=25,
        pixels_per_meter=4):

        super().__init__()
        self.point_net = DynamicPointNet(num_input, num_features)

        self.nx = (max_x-min_x) * pixels_per_meter
        self.ny = (max_y-min_y) * pixels_per_meter
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.pixels_per_meter = pixels_per_meter

    def decorate(self, points, unique_coords, inverse_indices):
        dtype = points.dtype
        x_centers = unique_coords[inverse_indices][:, 2:3].to(dtype) / self.pixels_per_meter + self.min_x
        y_centers = unique_coords[inverse_indices][:, 1:2].to(dtype) / self.pixels_per_meter + self.min_y

        xyz = points[:, :3]

        points_cluster = xyz - scatter_mean(xyz, inverse_indices, dim=0)[inverse_indices]

        points_xp = xyz[:, :1] - x_centers
        points_yp = xyz[:, 1:2] - y_centers

        features = torch.cat([points, points_cluster, points_xp, points_yp], dim=-1)
        return features

    def grid_locations(self, points):
        keep = (points[:, 0] >= self.min_x) & (points[:, 0] < self.max_x) & \
            (points[:, 1] >= self.min_y) & (points[:, 1] < self.max_y)
        points = points[keep, :]

        coords = (points[:, [0, 1]] - torch.tensor([self.min_x, self.min_y],
            device=points.device)) * self.pixels_per_meter
        coords = coords.long()

        return points, coords

    def pillar_generation(self, points, coords):
        unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)
        #decorated_points = self.decorate(points, unique_coords, inverse_indices)
        decorated_points = custom_decorate(points, unique_coords, inverse_indices)
        return decorated_points, unique_coords, inverse_indices

    def scatter_points(self, features, coords, batch_size ):
        #canvas = torch.zeros(batch_size, features.shape[1], self.ny, self.nx, dtype=features.dtype, device=features.device)
        canvas = custom_zero_tensor(batch_size, features.shape[1], self.ny, self.nx, dtype=features.dtype, device=features.device)
        #canvas[coords[:, 0], :, torch.clamp(self.ny-1-coords[:, 1],0,self.ny-1), torch.clamp(coords[:, 2],0,self.nx-1)] = features
        canvas = custom_scatter_points(features, coords, batch_size, self.ny, self.nx)
        return canvas
    
    

        return coords, filtered_points
    
    def forward(self, lidar_list, num_points):

        batch_size = len(lidar_list)
        with torch.no_grad():
            coords = []
            filtered_points = []
            '''
            for batch_id, points in enumerate(lidar_list):
                points = points[:num_points[batch_id]]
                #points[:, 1] = -points[:, 1]
                points, grid_yx= self.grid_locations(points)

                # batch indices 
                grid_byx = torch.nn.functional.pad(grid_yx,
                    (1, 0), mode='constant', value=batch_id)

                coords.append(grid_byx)
                filtered_points.append(points)

            '''
            coords, filtered_points = process_lidar_batch(lidar_list, num_points)
            # batch_size, grid_y, grid_x 
            #coords = torch.cat(coords, dim=0)
            #filtered_points = torch.cat(filtered_points, dim=0)
            coords, filtered_points = custom_cat(coords, filtered_points)
            decorated_points, unique_coords, inverse_indices = self.pillar_generation(filtered_points, coords)

        features = self.point_net(decorated_points, inverse_indices)

        return self.scatter_points(features, unique_coords, batch_size)

class ConvBackbone(nn.Module):
    def __init__(self, num_feature=64, norm_cfg={'eps': 1e-3, 'momentum': 0.01}):
        """
        Original PointPillar Backbone
        TODO: Write this better using for loops...
        """

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_feature,num_feature,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
            nn.Conv2d(num_feature,num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
            nn.Conv2d(num_feature,num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
            nn.Conv2d(num_feature,num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_feature, **norm_cfg),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feature,2*num_feature,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_feature,2*num_feature,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
            nn.Conv2d(2*num_feature,2*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_feature,2*num_feature,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(2*num_feature,2*num_feature,4,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(2*num_feature,2*num_feature,4,4,1,2,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*num_feature, **norm_cfg),
        )

        self.output = nn.Sequential(
            nn.Conv2d(6*num_feature,6*num_feature,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6*num_feature, **norm_cfg),
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        u1 = self.upconv1(x1)
        u2 = self.upconv2(x2)
        u3 = self.upconv3(x3)

        u = torch.cat([u1,u2,u3], dim=1)
        u = self.output(u)
        return u


@register_model
def pointpillar(**kwargs):
    return PointPillarNet(**kwargs)


@register_model
def pointpillar_conv(**kwargs):
    return ConvBackbone(**kwargs)
