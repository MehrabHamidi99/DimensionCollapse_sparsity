from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering, BisectingKMeans
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm
import numpy as np
import itertools
from sklearn.linear_model import RANSACRegressor

import torch
from DataLoader import get_simple_data_loader, get_data_loader
from ForwardPass import hook_forward_train


def spike_detection_nd(points, max_hyperplanes=30, min_points_for_hyperplane=100, residual_threshold=0.5, merge_threshold=0.01):
    residual_threshold *= (points.shape[1] - 1)
    # merge_threshold *= (points.shape[1] - 1)
    def fit_hyperplane_ransac(points):
        n_dims = points.shape[1]
        dependent_dim = np.random.randint(n_dims)
        
        X = np.delete(points, dependent_dim, axis=1)
        y = points[:, dependent_dim]

        
        ransac = RANSACRegressor(LinearRegression(), 
                                max_trials=1000, 
                                min_samples=n_dims,
                                residual_threshold=residual_threshold,
                                stop_probability=0.99)
        ransac.fit(X, y)
        return ransac, dependent_dim

    def get_hyperplane_eq(ransac, dependent_dim, n_dims):
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_
        
        full_coef = np.zeros(n_dims)
        full_coef[:dependent_dim] = coef[:dependent_dim]
        full_coef[dependent_dim+1:] = coef[dependent_dim:]
        full_coef[dependent_dim] = -1
        
        return full_coef, intercept

    def point_to_hyperplane_distance(point, coef, intercept):
        return np.abs(np.dot(coef, point) + intercept) / np.linalg.norm(coef)

    def hyperplane_distance(coef1, intercept1, coef2, intercept2):
        # Calculate the angle between the normal vectors of the hyperplanes
        cos_angle = np.dot(coef1, coef2) / (np.linalg.norm(coef1) * np.linalg.norm(coef2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Calculate the distance between the hyperplanes at the origin
        d = np.abs(intercept1 - intercept2) / np.linalg.norm(coef1)
        
        return angle, d

    n_dims = points.shape[1]
    hyperplanes = []
    remaining_points = points.copy()
    assigned_points = []
    total_error = 0

    for _ in range(max_hyperplanes):
        if len(remaining_points) < min_points_for_hyperplane:
            break
        
        try:
            ransac, dependent_dim = fit_hyperplane_ransac(remaining_points)
        except:
            return [], 0, []
        coef, intercept = get_hyperplane_eq(ransac, dependent_dim, n_dims)
        
        inlier_mask = ransac.inlier_mask_
        inlier_points = remaining_points[inlier_mask]
        
        hyperplane_error = sum(point_to_hyperplane_distance(p, coef, intercept) for p in inlier_points)
        total_error += hyperplane_error
        
        hyperplanes.append((coef, intercept, len(inlier_points), hyperplane_error))
        assigned_points.append(inlier_points)
        
        remaining_points = remaining_points[~inlier_mask]
    
    if len(hyperplanes) == 0:
        return [], 0, []

    # Assign remaining points to the nearest hyperplane
    for point in remaining_points:
        distances = [point_to_hyperplane_distance(point, coef, intercept) for coef, intercept, _, _ in hyperplanes]
        nearest_hyperplane_index = np.argmin(distances)
        total_error += distances[nearest_hyperplane_index]
        assigned_points[nearest_hyperplane_index] = np.vstack([assigned_points[nearest_hyperplane_index], point])

    # Merge close hyperplanes
    i = 0
    while i < len(hyperplanes):
        j = i + 1
        while j < len(hyperplanes):
            angle, dist = hyperplane_distance(hyperplanes[i][0], hyperplanes[i][1], hyperplanes[j][0], hyperplanes[j][1])
            # if angle < merge_threshold and dist < merge_threshold:
            if angle < merge_threshold:
                # Merge hyperplanes
                new_coef = (hyperplanes[i][0] + hyperplanes[j][0]) / 2
                new_intercept = (hyperplanes[i][1] + hyperplanes[j][1]) / 2
                new_inliers = hyperplanes[i][2] + hyperplanes[j][2]
                new_error = hyperplanes[i][3] + hyperplanes[j][3]
                new_points = np.vstack([assigned_points[i], assigned_points[j]])
                
                hyperplanes[i] = (new_coef, new_intercept, new_inliers, new_error)
                assigned_points[i] = new_points
                
                # Remove the merged hyperplane
                del hyperplanes[j]
                del assigned_points[j]
            else:
                j += 1
        i += 1

    # Sort hyperplanes by number of inliers (descending)
    hyperplanes = sorted(zip(hyperplanes, assigned_points), key=lambda x: x[0][2], reverse=True)
    hyperplanes, assigned_points = zip(*hyperplanes)

    return hyperplanes, total_error, assigned_points


def assign_points_to_hyperplanes(points, hyperplanes):
        """
        Efficiently assign points to the closest hyperplane and calculate the error.

        Parameters:
        points: numpy array of shape (n_points, n_dimensions)
        hyperplanes: list of tuples (coef, intercept), where coef is a numpy array

        Returns:
        assignments: numpy array of shape (n_points,) containing hyperplane indices
        total_error: float, sum of distances from points to their assigned hyperplanes
        hyperplane_errors: list of errors for each hyperplane
        """
        n_points, n_dims = points.shape
        n_hyperplanes = len(hyperplanes)

        # Prepare hyperplane coefficients and intercepts
        coefs = torch.Tensor([h[0] for h in hyperplanes]).to(points.device)
        intercepts = torch.Tensor([h[1] for h in hyperplanes]).to(points.device)

        # Calculate distances from each point to each hyperplane
        # Using broadcasting to avoid explicit loops
        distances = torch.abs(torch.matmul(points, coefs.T) + intercepts) / torch.linalg.norm(coefs, dim=1)

        # Find the closest hyperplane for each point
        assignments = torch.argmin(distances, dim=1)

        # Calculate the total error
        total_error = torch.sum(torch.min(distances, dim=1)[0])

        # Calculate error for each hyperplane
        hyperplane_errors = [torch.sum(distances[assignments == i, i]) for i in range(n_hyperplanes)]

        return assignments, total_error, hyperplane_errors


def spike_error(relu_outputs, labels, start_layer, device, true_labels=None):
    
    loss = torch.tensor(0, dtype=torch.float32).to(device)


    for i in range(start_layer, len(relu_outputs)):
        total_error = 0
        for each_label in torch.unique(labels):
            if true_labels is not None:
                class_data_indices = torch.where(true_labels == each_label)[0]
            else:
                class_data_indices = torch.where(labels == each_label)[0]
            points = relu_outputs[i][class_data_indices].clone()

            if points.shape[0] == 0:
                continue
            if points.shape[0] < points.shape[1]:
                continue
            
            class_data_indices = torch.where(labels == each_label)[0]
            points = relu_outputs[i][class_data_indices].clone()

            detected_hyperplanes, _, _ = spike_detection_nd(points.detach().cpu().numpy())
            if len(detected_hyperplanes) > 0:
                _, total_error_, _ = assign_points_to_hyperplanes(points, detected_hyperplanes)
                total_error += ((total_error_ / points.shape[0]))
    
        # Convert total_error to a differentiable tensor
        # loss += ((total_error / len(torch.unique(labels))) * (i - start_layer + 1))
        loss += (total_error / len(torch.unique(labels)))
    
    return loss

def spike_loss(traning_data, training_labels, optimizer, feature_extractor, device, this_batch_size=40000):

    loss_tracker = torch.tensor(0, dtype=torch.float32).to(device)
    i = 0

    data_loader = get_data_loader(traning_data, training_labels, batch_size=this_batch_size, shuffle=False)

    for sample, target in data_loader:
        sample = sample.to(device)
        target = target.to(device)
        output, relu_outputs = hook_forward_train(feature_extractor, sample)
        pred = torch.exp(output).argmax(dim=1, keepdim=True)
        pred = pred.view_as(target)
        loss = spike_error(relu_outputs, pred, 5, device, true_labels=target).requires_grad_(True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker += loss.detach()

        i += 1
    
    # loss /= i

    return loss_tracker / i

    # loss = torch.tensor(0, dtype=torch.float32).to(device)
    # i = 0

    # data_loader = get_data_loader(traning_data, training_labels, batch_size=this_batch_size, shuffle=False)

    # for sample, target in data_loader:
    #     sample = sample.to(device)
    #     relu_outputs = hook_forward(feature_extractor, sample)
    #     loss += spike_error(relu_outputs, target, 3, device)
    #     i += 1
    
    # # loss /= i

    # return loss.requires_grad_(True)



