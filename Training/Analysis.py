from utils import *
from utils_plotting import *
from Models.FeatureExtractor import *
from Models.ForwardPass import *
import ctypes
import gc

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

@torch.no_grad
def fixed_model_batch_analysis(model, samples, labels, device, save_path, model_status, batch_size=10000):

    FIRST_BATCH = None

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
        'labels': [[]] # init layer, softmax
    }

    batch_labels = []

    feature_extractor = ReluExtractor(model, device=device)

    this_batch_size = min(batch_size, samples.shape[0])

    data_loader = get_data_loader(samples, labels, batch_size=this_batch_size, shuffle=False)

    FIRST_BATCH = True

    results_dict['representations'] = []
    results_dict['representations'].append(samples)

    for sample, label in data_loader:
        sample = sample.to(device)
        label = label.to(device)

        with torch.no_grad():

            output = model(sample)
            pred = torch.argmax(output, dim=1)
            # pred = torch.squeeze(pred).detach().cpu().numpy().tolist()
        # batch_labels.extend(pred)
        # batch_labels.extend(label.detach().cpu().numpy().tolist())

        sample_ = sample.view(sample.shape[0], -1)

        if FIRST_BATCH:
            covariance_matrix = _covar_calc(sample_)
            results_dict = batch_projectional_analysis(covariance_matrix, sample_, results_dict, FIRST_BATCH, this_index=0, preds=pred)

            results_dict = _add_covar(results_dict, sample_, covariance_matrix)
        else:
            results_dict = _update_covar(results_dict, sample_, covariance_matrix, 0)

        # relu_outputs = hook_forward(feature_extractor, sample, label, device)
        relu_outputs = hook_forward(feature_extractor, sample)

        # ـ, relu_outputs = feature_extractor(samples)

        results_dict, FIRST_BATCH, new_labels = _on_the_go_analysis(results_dict, relu_outputs, FIRST_BATCH, sample_, pred)
        results_dict['labels'][0].extend(label.detach().cpu().numpy())
        results_dict['this_batch_size'] = results_dict['this_batch_size'] + sample.shape[0]


    results_dict['display_matrix'] = generate_heatmap_from_the_activation_list(results_dict['activations'])


    for i in range(len(results_dict['covar_matrix'])):
        results_dict = covariance_matrix_additional_and_projectional(results_dict['covar_matrix'][i], results_dict, device)

    plotting_actions(results_dict, num=samples.shape[0], this_path=save_path, arch=model_status)

    plot_gifs(results_dict, this_path=save_path, num=samples.shape[0], costume_range=100, pre_path=save_path, eigenvectors=np.array(results_dict['eigenvectors'], dtype=object), labels=results_dict['labels'])
    
    # Empty the GPU cache
    del relu_outputs, data_loader, feature_extractor
    gc.collect()

    torch.cuda.empty_cache()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    return results_dict


@torch.no_grad
def _covar_calc(this_data):
    new_data = this_data

    batch_data_centered = new_data - torch.mean(new_data, axis=0)

    # Compute covariance matrix for this batch
    return torch.matmul(batch_data_centered.T, batch_data_centered) / (batch_data_centered.shape[0] - 1)

@torch.no_grad
def _on_the_go_analysis(result_dict, relu_outputs_batch, FIRST_BATCH, sample, preds):

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
        covariance_matrix = _covar_calc(sample)
        result_dict = batch_projectional_analysis(covariance_matrix, sample, result_dict, FIRST_BATCH, this_index=0, preds=preds)

    new_labels = preds.detach().cpu().numpy().tolist()

    for i in range(len(relu_outputs_batch)):

        # layer_act = relu_outputs_batch[i].detach().cpu().numpy()
        layer_act = relu_outputs_batch[i]


        if len(layer_act.shape) == 3:
            layer_act = torch.mean(layer_act, dim=1).view(layer_act.shape[0], -1)
        if len(layer_act.shape) > 3:
            layer_act = layer_act.view(layer_act.shape[0], -1)

        # result_dict['labels'][i + 1].extend(preds.detach().cpu().numpy().tolist())

        non_zero = torch.sum((layer_act > 0).detach().cpu().int(), axis=0) # type: ignore # D

        new_results['activations'].append(non_zero.numpy())
        new_results['non_zero_activations_layer_wise'].append(torch.sum((non_zero > 0).detach().cpu().int()).numpy())

        cell_dim = torch.sum((layer_act > 0).detach().cpu().int(), axis=1) # type: ignore # B

        new_results['cell_dimensions'].append(cell_dim)
        new_results['batch_cell_dimensions'].append(torch.min(cell_dim).numpy())

        covariance_matrix = _covar_calc(layer_act)
        result_dict = batch_projectional_analysis(covariance_matrix, layer_act, result_dict, first_batch=FIRST_BATCH, this_index=i + 1)

        if FIRST_BATCH:
            result_dict['representations'].append(layer_act)
            result_dict = _add_covar(result_dict, relu_outputs_batch[i], covariance_matrix)
        else:
            result_dict['representations'][i + 1] = torch.cat([result_dict['representations'][i + 1], layer_act])
            result_dict = _update_covar(result_dict, relu_outputs_batch[i], covariance_matrix, i + 1)

    if FIRST_BATCH:
        FIRST_BATCH = False
    return merge_results_batch(result_dict, new_results), FIRST_BATCH, new_labels

@torch.no_grad
def _update_covar(result_dict, this_data, covar_matrix, i):

    # Store covariance matrix and batch size
    result_dict['covar_matrix'][i] = (result_dict['covar_matrix'][i] * result_dict['this_batch_size'] +\
            covar_matrix * this_data.shape[0]) / (result_dict['this_batch_size'] + this_data.shape[0])
    return result_dict

@torch.no_grad
def _add_covar(result_dict, this_data, covar_matrix):

    result_dict['covar_matrix'].append(covar_matrix)

    return result_dict


def fixed_model_batch_analysis_one_batch(model, samples, labels, device, save_path, model_status):

    FIRST_BATCH = None

    def on_the_go_analysis(result_dict, relu_outputs_batch, FIRST_BATCH, sample, labels):

        # Collect activations and labels per layer
        # For the input layer (layer 0)
        layer_act_input = sample.detach().cpu().numpy()
        result_dict['layer_activations'][0].append(layer_act_input)
        result_dict['layer_labels'][0].extend(labels.detach().cpu().numpy().tolist())

        for i in range(len(relu_outputs_batch)):
            layer_act = relu_outputs_batch[i].detach().cpu().numpy()
            result_dict['layer_activations'][i + 1].append(layer_act)
            result_dict['layer_labels'][i + 1].extend(labels.detach().cpu().numpy().tolist())

        return result_dict
    
    # Initialize result_dict with lists to collect activations and labels
    results_dict = {
        'layer_activations': [[] for _ in range(len(model.layer_list) + 1 + 1)],  # Including input layer
        'layer_labels': [[] for _ in range(len(model.layer_list) + 1 + 1)],

        # 'activations': [],
        # 'non_zero_activations_layer_wise' : [],
        # 'cell_dimensions': [],
        # 'batch_cell_dimensions': [],
        # 'nonzero_eigenvalues_count': [],
        # 'eigenvalues': [],
        # 'stable_rank': [],
        # 'simple_spherical_mean_width': [],
        # 'spherical_mean_width_v2': [],
        # 'norms': [],
        # 'eigenvectors': [],
        # 'pca_2': [], 
        # 'pca_3': [],
        # 'random_2': [],
        # 'random_3': [],
    }

    # Rest of your code remains mostly the same, with adjustments to pass labels
    feature_extractor = ReluExtractor(model, device=device)
    this_batch_size = min(10000, samples.shape[0])
    # this_batch_size = samples.shape[0]
    data_loader = get_data_loader(samples, labels, batch_size=this_batch_size, shuffle=False)

    for sample, label in data_loader:
        sample = sample.to(device)
        label = label.to(device)

        output = model(sample).detach().cpu().numpy()
        # pred = torch.argmax(output, dim=1)  # Not needed anymore

        # relu_outputs = hook_forward(feature_extractor, sample, label, device)
        relu_outputs = hook_forward(feature_extractor, sample)

        results_dict = on_the_go_analysis(results_dict, relu_outputs, FIRST_BATCH, sample, label)


    # After processing all batches, concatenate activations and labels per layer
    for i in range(len(results_dict['layer_activations'])):
        results_dict['layer_activations'][i] = np.concatenate(results_dict['layer_activations'][i], axis=0)
        # Labels are already extended as lists

    final_results_dict = all_analysis_for_hook_engine(results_dict['layer_activations'][1:], results_dict['layer_activations'][0])
    final_results_dict['layer_labels'] = results_dict['layer_labels']
    final_results_dict['layer_activations'] = results_dict['layer_activations']

    del results_dict

    # Perform PCA and other analyses after aggregation
    final_results_dict = perform_pca_and_analyses(final_results_dict, device)

    plotting_actions(final_results_dict, num=samples.shape[0], this_path=save_path, arch=model_status)

    # Plotting
    plot_gifs(final_results_dict, this_path=save_path, num=samples.shape[0], pre_path=save_path, labels=final_results_dict['layer_labels'])
    del final_results_dict
    del relu_outputs, data_loader