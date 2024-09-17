from utils import *
from FeatureExtractor import *
from ForwardPass import *
from FeatureExtractor import *

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

def merge_results_batch(results_dict, batch_result_dict, sum_ones=['activations', 'non_zero_activations_layer_wise', 'cell_dimensions', 'batch_cell_dimensions']):
    for k, v in results_dict.items():
        if k in sum_ones:
            try:
                if len(v) == 0:
                    results_dict[k] = v + batch_result_dict[k]
                else:
                    for j in range(len(v)):
                        results_dict[k][j] = v[j] + batch_result_dict[k][j]             
            except:
                pass 
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



def fixed_model_batch_analysis(model, samples, labels, device, save_path, model_status):

    FIRST_BATCH = None

    def covar_calc(this_data):
        batch_data_centered = this_data - torch.mean(this_data, axis=0)

        # Compute covariance matrix for this batch

        return torch.matmul(batch_data_centered.T, batch_data_centered) / (batch_data_centered.shape[0] - 1)



    def on_the_go_analysis(result_dict, relu_outputs_batch, FIRST_BATCH, sample, preds):

        new_results = {
            'activations': [],
            'non_zero_activations_layer_wise' : [],
            'cell_dimensions': [],
            'batch_cell_dimensions': [],
            'nonzero_eigenvalues_count': [],
            # 'spherical_mean_width_v2': [],
            # 'norms': [],
            # 'pca_2': [], 
            # 'pca_3': [],
            # 'random_2': [],
            # 'random_3': [],
        }
        if not FIRST_BATCH:
            covariance_matrix = covar_calc(sample)
            result_dict = batch_projectional_analysis(covariance_matrix, sample.detach().cpu().numpy(), result_dict, FIRST_BATCH, this_index=0, preds=preds)

        for i in range(len(relu_outputs_batch)):

            layer_act = relu_outputs_batch[i].detach().cpu().numpy()

            non_zero = np.sum((layer_act > 0).astype(int), axis=0) # D

            new_results['activations'].append(non_zero)
            new_results['non_zero_activations_layer_wise'].append(np.sum((non_zero > 0).astype(int)))

            cell_dim = np.sum((layer_act > 0).astype(int), axis=1) # B

            new_results['cell_dimensions'].append(cell_dim)
            new_results['batch_cell_dimensions'].append(np.min(cell_dim))

            covariance_matrix = covar_calc(relu_outputs_batch[i])
            result_dict = batch_projectional_analysis(covariance_matrix, layer_act, result_dict, first_batch=FIRST_BATCH, this_index=i + 1)

            if FIRST_BATCH:
                result_dict = add_covar(result_dict, relu_outputs_batch[i], covariance_matrix)
            else:
                result_dict = update_covar(result_dict, relu_outputs_batch[i], covariance_matrix, i + 1)

        if FIRST_BATCH:
            FIRST_BATCH = False
        return merge_results_batch(result_dict, new_results), FIRST_BATCH

    def update_covar(result_dict, this_data, covar_matrix, i):

        # Store covariance matrix and batch size
        result_dict['covar_matrix'][i] = (result_dict['covar_matrix'][i] * result_dict['this_batch_size'] +\
              covar_matrix * this_data.shape[0]) / (result_dict['this_batch_size'] + this_data.shape[0])
        return result_dict


    def add_covar(result_dict, this_data, covar_matrix):

        result_dict['covar_matrix'].append(covar_matrix)

        return result_dict


    results_dict = {

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
        'random_3': [],

        'covar_matrix': [],
        'this_batch_size': 0,
        'labels': []
    }

    batch_labels = []

    feature_extractor = ReluExtractor(model, device=device)

    this_batch_size = min(10000, samples.shape[0])

    data_loader = get_data_loader(samples, labels, batch_size=this_batch_size)

    FIRST_BATCH = True

    for sample, label in data_loader:
        sample = sample.to(device)
        label = label.to(device)

        output = torch.exp(model(sample))
        pred = torch.argmax(output, dim=1)
        # pred = torch.squeeze(pred).detach().cpu().numpy().tolist()
        # batch_labels.extend(pred)
        batch_labels.extend(label.detach().cpu().numpy().tolist())

        if FIRST_BATCH:
            covariance_matrix = covar_calc(sample)
            results_dict = batch_projectional_analysis(covariance_matrix, sample.detach().cpu().numpy(), results_dict, FIRST_BATCH, this_index=0, preds=pred)

            results_dict = add_covar(results_dict, sample, covariance_matrix)
        else:
            results_dict = update_covar(results_dict, sample, covariance_matrix, 0)

        relu_outputs = hook_forward(feature_extractor, sample, label, device)
        # ـ, relu_outputs = feature_extractor(samples)

        results_dict, FIRST_BATCH = on_the_go_analysis(results_dict, relu_outputs, FIRST_BATCH, sample, pred)
        results_dict['this_batch_size'] = results_dict['this_batch_size'] + sample.shape[0]


    results_dict['display_matrix'] = generate_heatmap_from_the_activation_list(results_dict['activations'])


    for i in range(len(results_dict['covar_matrix'])):
        results_dict = covariance_matrix_additional_and_projectional(results_dict['covar_matrix'][i], results_dict, device)

    plotting_actions(results_dict, num=samples.shape[0], this_path=save_path, arch=model_status)

    plot_gifs(results_dict, this_path=save_path, num=samples.shape[0], costume_range=100, pre_path=save_path, eigenvectors=np.array(results_dict['eigenvectors'], dtype=object), labels=batch_labels)
  
