from utils import *
from DataGenerator import *
from Models_normal import *
import torch.optim as optim
from Analysis import fixed_model_batch_analysis

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

def generate_data(mode, params={}, num=20000): 

    if mode == '2island':

        means = params.get('means', [[50, 50], [-50, 50]])
        variances = params.get('variances', [5, 5])
        
        # Generate data for each of the two components of class 1
        data1 = []
        for mean, variance in zip(means, variances):
            data = create_random_data_normal_dist(input_dimension=2, num=int(num/2), loc=mean, scale=np.sqrt(variance))
            data1.append(data)

        # Combine both components of class 1
        data1 = np.vstack(data1)

        # Parameters for the second Gaussian distribution (negative label)
        mean2 = params.get('mean2', [0, 0])  # Mean of the second class (negative label)
        variance2 = params.get('variance2', 500)  # Variance of the second class (much larger)
        data2 = create_random_data_normal_dist(input_dimension=2, num=num, loc=mean2, scale=np.sqrt(variance2))

        # Combine the datasets and create labels
        X = np.vstack((data1, data2))
        y = np.array([1] * num + [0] * num)  # Positive labels for the first class and negative for the second

        return X, y 

    elif mode == '4island':
        # Extract parameters from the dictionary
        means = params.get('means', [[50, 50], [50, -50], [-50, 50], [-50, -50]])
        variances = params.get('variances', [0.25, 0.25, 0.25, 0.25])

        # Generate data for each of the four components of class 1
        data1 = []
        for mean, variance in zip(means, variances):
            data = create_random_data_normal_dist(input_dimension=2, num=int(num/4), loc=mean, scale=np.sqrt(variance))
            data1.append(data)

        # Combine all components of class 1
        data1 = np.vstack(data1)

        # Parameters for the second Gaussian distribution (negative label)
        mean2 = params.get('mean2', [0, 0])  # Mean of the second class (negative label)
        variance2 = params.get('variance2', 400)  # Variance of the second class (much larger)
        data2 = create_random_data_normal_dist(input_dimension=2, num=num, loc=mean2, scale=np.sqrt(variance2))

        # Combine the datasets and create labels
        X = np.vstack((data1, data2))
        y = np.array([1] * num + [0] * num)  # Positive labels for the first class and negative for the second

        return X, y
    
    elif mode == 'inside_circles':
        # Extract parameters from the dictionary
        radius_small = params.get('radius_small', 10)
        radius_middle = params.get('radius_middle', 20)
        radius_large = params.get('radius_large', 30)

        # Generate data for the smallest circle (belongs to class 1)
        angle_small = np.random.uniform(0, 2 * np.pi, int(num / 3))
        data_small = np.c_[radius_small * np.cos(angle_small), radius_small * np.sin(angle_small)]

        # Generate data for the middle circle (belongs to class 0)
        angle_middle = np.random.uniform(0, 2 * np.pi, int(num / 3))
        data_middle = np.c_[radius_middle * np.cos(angle_middle), radius_middle * np.sin(angle_middle)]

        # Generate data for the largest circle (belongs to class 1)
        angle_large = np.random.uniform(0, 2 * np.pi, int(num / 3))
        data_large = np.c_[radius_large * np.cos(angle_large), radius_large * np.sin(angle_large)]

        # Combine the datasets for class 1 (smallest and largest circles)
        data_class1 = np.vstack((data_small, data_large))
        # Class 0 is the middle circle
        data_class0 = data_middle

        # Combine the datasets and create labels
        X = np.vstack((data_class1, data_class0))
        y = np.array([1] * len(data_class1) + [0] * len(data_class0))  # Class 1 for smallest and largest circles, class 0 for the middle circle

        return X, y
    
    elif mode == 'donut_cloud':
        # Extract parameters from the dictionary
        inner_radius_small = params.get('inner_radius_small', 10)
        outer_radius_small = params.get('outer_radius_small', 15)
        inner_radius_middle = params.get('inner_radius_middle', 20)
        outer_radius_middle = params.get('outer_radius_middle', 25)
        inner_radius_large = params.get('inner_radius_large', 30)
        outer_radius_large = params.get('outer_radius_large', 35)
        variance = params.get('variance', 1.0)

        # Generate random angles and radii within the donut shape for the smallest donut (class 1)
        angles_small = np.random.uniform(0, 2 * np.pi, int(num / 3))
        radii_small = np.random.uniform(inner_radius_small, outer_radius_small, int(num / 3))
        radii_noise_small = np.random.normal(0, variance, int(num / 3))
        final_radii_small = radii_small + radii_noise_small
        data_small_donut = np.c_[final_radii_small * np.cos(angles_small), final_radii_small * np.sin(angles_small)]

        # Generate random angles and radii within the donut shape for the middle donut (class 0)
        angles_middle = np.random.uniform(0, 2 * np.pi, int(num / 3))
        radii_middle = np.random.uniform(inner_radius_middle, outer_radius_middle, int(num / 3))
        radii_noise_middle = np.random.normal(0, variance, int(num / 3))
        final_radii_middle = radii_middle + radii_noise_middle
        data_middle_donut = np.c_[final_radii_middle * np.cos(angles_middle), final_radii_middle * np.sin(angles_middle)]

        # Generate random angles and radii within the donut shape for the largest donut (class 1)
        angles_large = np.random.uniform(0, 2 * np.pi, int(num / 3))
        radii_large = np.random.uniform(inner_radius_large, outer_radius_large, int(num / 3))
        radii_noise_large = np.random.normal(0, variance, int(num / 3))
        final_radii_large = radii_large + radii_noise_large
        data_large_donut = np.c_[final_radii_large * np.cos(angles_large), final_radii_large * np.sin(angles_large)]

        # Combine the datasets for class 1 (smallest and largest donuts)
        data_class1 = np.vstack((data_small_donut, data_large_donut))
        # Class 0 is the middle donut
        data_class0 = data_middle_donut

        # Combine the datasets and create labels
        X = np.vstack((data_class1, data_class0))
        y = np.array([1] * len(data_class1) + [0] * len(data_class0))  # Class 1 for smallest and largest donuts, class 0 for the middle donut

        return X, y
    
    elif mode == 'four_donuts':
        # Extract parameters from the dictionary
        radii = params.get('radii', [10, 15, 20, 25])
        variance = params.get('variance', 1.0)

        # Generate random angles and radii within the donut shape for each donut
        data_class1 = []
        data_class0 = []

        for i, radius in enumerate(radii):
            angles = np.random.uniform(0, 2 * np.pi, int(num / 4))
            radii_noise = np.random.normal(0, variance, int(num / 4))
            final_radii = radius + radii_noise
            data_donut = np.c_[final_radii * np.cos(angles), final_radii * np.sin(angles)]
            
            # Assign classes: largest and third largest to class 1, second and fourth largest to class 0
            if i in [0, 2]:
                data_class1.append(data_donut)
            else:
                data_class0.append(data_donut)

        # Combine the datasets for each class
        data_class1 = np.vstack(data_class1)
        data_class0 = np.vstack(data_class0)

        # Combine the datasets and create labels
        X = np.vstack((data_class1, data_class0))
        y = np.array([1] * len(data_class1) + [0] * len(data_class0))  # Class 1 for largest and third largest donuts, class 0 for others

        return X, y
    
    elif mode == 'donut_and_line':
        # Extract parameters from the dictionary
        inner_radius_small = params.get('inner_radius_small', 10)
        outer_radius_small = params.get('outer_radius_small', 15)
        inner_radius_middle = params.get('inner_radius_middle', 20)
        outer_radius_middle = params.get('outer_radius_middle', 25)
        inner_radius_large = params.get('inner_radius_large', 30)
        outer_radius_large = params.get('outer_radius_large', 35)
        variance = params.get('variance', 1.0)

        # Generate random angles and radii within the donut shape for the smallest donut (class 1)
        angles_small = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_small = np.random.uniform(inner_radius_small, outer_radius_small, int(num / 4))
        radii_noise_small = np.random.normal(0, variance, int(num / 4))
        final_radii_small = radii_small + radii_noise_small
        data_small_donut = np.c_[final_radii_small * np.cos(angles_small), final_radii_small * np.sin(angles_small)]

        # Generate random angles and radii within the donut shape for the middle donut (class 0)
        angles_middle = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_middle = np.random.uniform(inner_radius_middle, outer_radius_middle, int(num / 4))
        radii_noise_middle = np.random.normal(0, variance, int(num / 4))
        final_radii_middle = radii_middle + radii_noise_middle
        data_middle_donut = np.c_[final_radii_middle * np.cos(angles_middle), final_radii_middle * np.sin(angles_middle)]

        # Generate random angles and radii within the donut shape for the largest donut (class 1)
        angles_large = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_large = np.random.uniform(inner_radius_large, outer_radius_large, int(num / 4))
        radii_noise_large = np.random.normal(0, variance, int(num / 4))
        final_radii_large = radii_large + radii_noise_large
        data_large_donut = np.c_[final_radii_large * np.cos(angles_large), final_radii_large * np.sin(angles_large)]

        # Combine the datasets for class 1 (smallest and largest donuts)
        data_class1 = np.vstack((data_small_donut, data_large_donut))
        # Class 0 is the middle donut
        data_class0 = data_middle_donut

        # Generate data points around a random line intersecting all donuts (belongs to class 1)
        slope = np.random.uniform(-1, 1)
        intercept = np.random.uniform(-outer_radius_large, outer_radius_large)
        line_x = np.linspace(-outer_radius_large, outer_radius_large, int(num / 4))
        line_y = slope * line_x + intercept
        line_noise_x = np.random.normal(0, variance, int(num / 4))
        line_noise_y = np.random.normal(0, variance, int(num / 4))
        data_line = np.c_[line_x + line_noise_x, line_y + line_noise_y]

        # Filter out points that intersect with other classes
        distances_middle = np.sqrt((data_line[:, 0] - 0) ** 2 + (data_line[:, 1] - 0) ** 2)
        mask = (distances_middle < inner_radius_middle) | (distances_middle > outer_radius_middle)
        data_line_filtered = data_line[mask]

        # Combine line data with class 1
        data_class0 = np.vstack((data_class0, data_line_filtered))

        # Combine the datasets and create labels
        X = np.vstack((data_class1, data_class0))
        y = np.array([1] * len(data_class1) + [0] * len(data_class0))  # Class 1 for smallest and largest donuts and line, class 0 for middle donut

        return X, y
    
    elif mode == 'donut_and_bounded_line':
        # Extract parameters from the dictionary
        inner_radius_small = params.get('inner_radius_small', 10)
        outer_radius_small = params.get('outer_radius_small', 15)
        inner_radius_middle = params.get('inner_radius_middle', 20)
        outer_radius_middle = params.get('outer_radius_middle', 25)
        inner_radius_large = params.get('inner_radius_large', 30)
        outer_radius_large = params.get('outer_radius_large', 35)
        variance = params.get('variance', 1.0)

        # Generate random angles and radii within the donut shape for the smallest donut (class 1)
        angles_small = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_small = np.random.uniform(inner_radius_small, outer_radius_small, int(num / 4))
        radii_noise_small = np.random.normal(0, variance, int(num / 4))
        final_radii_small = radii_small + radii_noise_small
        data_small_donut = np.c_[final_radii_small * np.cos(angles_small), final_radii_small * np.sin(angles_small)]

        # Generate random angles and radii within the donut shape for the middle donut (class 0)
        angles_middle = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_middle = np.random.uniform(inner_radius_middle, outer_radius_middle, int(num / 4))
        radii_noise_middle = np.random.normal(0, variance, int(num / 4))
        final_radii_middle = radii_middle + radii_noise_middle
        data_middle_donut = np.c_[final_radii_middle * np.cos(angles_middle), final_radii_middle * np.sin(angles_middle)]

        # Generate random angles and radii within the donut shape for the largest donut (class 1)
        angles_large = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_large = np.random.uniform(inner_radius_large, outer_radius_large, int(num / 4))
        radii_noise_large = np.random.normal(0, variance, int(num / 4))
        final_radii_large = radii_large + radii_noise_large
        data_large_donut = np.c_[final_radii_large * np.cos(angles_large), final_radii_large * np.sin(angles_large)]

        # Combine the datasets for class 1 (smallest and largest donuts)
        data_class1 = np.vstack((data_small_donut, data_large_donut))
        # Class 0 is the middle donut
        data_class0 = data_middle_donut

        # Generate data points around a random line fully inside the largest donut (belongs to class 1)
        slope = np.random.uniform(-1, 1)
        intercept = np.random.uniform(-inner_radius_large, inner_radius_large)
        line_x = np.linspace(-inner_radius_large, inner_radius_large, int(num / 4))
        line_y = slope * line_x + intercept
        line_noise_x = np.random.normal(0, variance, int(num / 4))
        line_noise_y = np.random.normal(0, variance, int(num / 4))
        data_line = np.c_[line_x + line_noise_x, line_y + line_noise_y]

        # Filter out points that are outside the inner boundary of the largest donut
        distances = np.sqrt(data_line[:, 0] ** 2 + data_line[:, 1] ** 2)
        mask = distances <= inner_radius_large
        data_line_bounded = data_line[mask]

        # Combine line data with class 1
        data_class1 = np.vstack((data_class1, data_line_bounded))

        # Combine the datasets and create labels
        X = np.vstack((data_class1, data_class0))
        y = np.array([1] * len(data_class1) + [0] * len(data_class0))  # Class 1 for smallest and largest donuts and bounded line, class 0 for middle donut

        return X, y
    
    elif mode == 'donuts_and_islands':
        
        inner_radius_small = params.get('inner_radius_small', 10)
        outer_radius_small = params.get('outer_radius_small', 15)
        inner_radius_large = params.get('inner_radius_large', 50)
        outer_radius_large = params.get('outer_radius_large', 60)
        island_means = params.get('island_means', [[25, 25], [25, -25], [-25, 25], [-25, -25]])
        island_variances = params.get('island_variances', [5, 5, 5, 5])
        variance = params.get('variance', 1.0)

        # Generate random angles and radii within the donut shape for the smallest donut (class 1)
        angles_small = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_small = np.random.uniform(inner_radius_small, outer_radius_small, int(num / 4))
        radii_noise_small = np.random.normal(0, variance, int(num / 4))
        final_radii_small = radii_small + radii_noise_small
        data_small_donut = np.c_[final_radii_small * np.cos(angles_small), final_radii_small * np.sin(angles_small)]

        # Generate random angles and radii within the donut shape for the large donut (class 1)
        angles_large = np.random.uniform(0, 2 * np.pi, int(num / 4))
        radii_large = np.random.uniform(inner_radius_large, outer_radius_large, int(num / 4))
        radii_noise_large = np.random.normal(0, variance, int(num / 4))
        final_radii_large = radii_large + radii_noise_large
        data_large_donut = np.c_[final_radii_large * np.cos(angles_large), final_radii_large * np.sin(angles_large)]

        # Combine the datasets for class 1 (small and large donuts)
        data_class1 = np.vstack((data_small_donut, data_large_donut))

        # Generate data for four islands completely between the two donuts (class 0)
        data_islands = []
        for mean, variance in zip(island_means, island_variances):
            data_island = create_random_data_normal_dist(input_dimension=2, num=int(num / 4), loc=mean, scale=np.sqrt(variance))
            data_islands.append(data_island)

        # Combine all island data for class 0
        data_class0 = np.vstack(data_islands)

        # Combine the datasets and create labels
        X = np.vstack((data_class1, data_class0))
        y = np.array([1] * len(data_class1) + [0] * len(data_class0))  # Class 1 for donuts, class 0 for islands

        return X, y
