from utils import *


def all_analysis_for_hook_engine(relu_outputs,  dataset):

    results = {
        'activations': [],
        'non_zero_activations_layer_wise' : [],
        'cell_dimensions': [],
        'batch_cell_dimensions': [],
        'nonzero_eigenvalues_count': [],
        'eigenvalues': [],
        'stable_rank': [],
        'simple_spherical_mean_width': [],
        'spherical_mean_width_v2': [],
        'norms': [],
        'eigenvectors': [],
        'pca_2': [], 
        'pca_3': [],
        'random_2': [],
        'random_3': []
    }

    results = additional_analysis_for_full_data(dataset, results)
    results = projection_analysis_for_full_data(dataset, results)


    for i in range(len(relu_outputs)):
        
        layer_act = relu_outputs[i]

        non_zero = np.sum((layer_act > 0).astype(int), axis=0) # D

        results['activations'].append(non_zero)
        results['non_zero_activations_layer_wise'].append(np.sum((non_zero > 0).astype(int)))

        cell_dim = np.sum((layer_act > 0).astype(int), axis=1) # B

        results['cell_dimensions'].append(cell_dim)
        results['batch_cell_dimensions'].append(np.min(cell_dim))

        results = additional_analysis_for_full_data(layer_act, results)
        results = projection_analysis_for_full_data(layer_act, results)

    results['display_matrix'] = generate_heatmap_from_the_activation_list(results['activations'])

    return results        

def merge_results(results_dict, new_result_dict):
    for k, v in results_dict.items():
        try:
            if k in new_result_dict:
                results_dict[k] = (v + new_result_dict[k]) / 2
        except:
            try:
                for j in range(len(v)):
                    results_dict[k][j] = (v[j] + new_result_dict[k][j]) / 2                
            except:
                for j in range(len(v)):
                    tmp_res = []
                    for l in range(len(v[j])):
                        tmp_res.append((v[j][l] + new_result_dict[k][j][l]) / 2)
                    results_dict[k][j] = tmp_res
    return results_dict


def generate_heatmap_from_the_activation_list(layer_activations):

    # Determine the maximum number of neurons in any layer
    max_neurons = max(len(layer) for layer in layer_activations)
    
    # Create a 2D array with NaN values to handle centering
    heatmap_data = np.full((max_neurons, len(layer_activations)), np.nan)

    # Fill the heatmap data, centering the neurons in the middle
    for i, layer in enumerate(layer_activations):
        num_neurons = len(layer)
        start_index = (max_neurons - num_neurons) // 2
        heatmap_data[start_index:start_index + num_neurons, i] = layer

    return heatmap_data
