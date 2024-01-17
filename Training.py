from Models import *
from utils import *
from DataGenerator import *
from ForwardPass import *


def train_model(model, train_loader, test_loader, base_path, train_x, val_x, val_loader=None, epochs=50, learning_rate=0.001, loss='mse'):
    '''
    Parameters
    model (torch.nn.Module): A PyTorch neural network model to be trained.
    train_loader (DataLoader): DataLoader containing the training dataset.
    val_loader (DataLoader, optional): DataLoader containing the validation dataset. If None, validation is skipped.
    epochs (int, optional): The number of epochs for which the model will be trained. Default is 50.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
    
    Returns
    val_analysis (list): A list containing the aggregated neuron activations for each epoch during validation.
    val_analysis_depth (list): A list containing the aggregated neuron activation ratios for each layer 
                                of the model for each epoch during validation.
    
    Functionality
    Trains the model using the Mean Squared Error loss and Adam optimizer.
    After each training epoch, evaluates the model on the validation dataset if provided.
    For each epoch during validation, performs an analysis of neuron activations using 
    the model's analysis_neurons_activations_depth_wise method.
    Collects and returns the neuron activation data across all epochs.
    
    Usage Example
    model = MLP_ReLU(n_in=10, layer_list=[20, 30, 1])
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    val_analysis, val_analysis_depth = train_model(model, train_loader, val_loader)
    
    Notes
    This function is particularly useful for understanding how the activations of neurons in 
    a neural network evolve over the course of training.
    The neuron activation analysis can provide insights into which neurons are 
    consistently active or inactive and how this pattern changes with each epoch.
    The function assumes that the model has a method analysis_neurons_activations_depth_wise 
    to perform the activation analysis, which should be implemented in the model class.
    This function is a comprehensive tool for training neural network models while simultaneously analyzing their behavior, 
    particularly useful for research and in-depth study of neural network dynamics.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    if loss == 'mse':
        criterion = F.mse_loss()

    # train_x = train_loader.dataset.dataset.data[train_loader.dataset.indices,:,:].to(device)
    # val_x = val_loader.dataset.dataset.data[val_loader.dataset.indices,:,:].to(device)
    # # test_x = test_loader.dataset.dataset[test_loader.dataset.indices,:,:].to(device)

    train_add = []
    train_eig = []
    val_add = []
    val_eig = []
    for epoch in tqdm(range(epochs)):
        model.not_extra()
        over_path = base_path + 'epoch_{}/'.format(str(epoch))
        if not os.path.isdir(over_path):
            os.makedirs(over_path)
        
        model.train()
        for x_batch, target in train_loader:
            x_batch, target = x_batch.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        correct = 0

        additive_act, eigen_count, _, _ = model.post_forward_neuron_activation_analysis()
        train_add += [additive_act]
        train_eig += [eigen_count]

        with torch.no_grad():
            model.not_extra()
            model.reset()
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            # additive_act, eigenvalues_count = whole_data_analysis_forward_pass(model, 'train', base_path, train_x)
            # train_eig += [eigenvalues_count]
            # additive_act, eigenvalues_count = whole_data_analysis_forward_pass(model, 'val', base_path, val_x)
            additive_act, eigenvalues_count, _, _  = model.post_forward_neuron_activation_analysis()
            val_add += [additive_act]
            val_eig += [eigenvalues_count]

        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Val loss: {val_loss:.4f}, Accuracy: {100. * correct / len(val_loader.dataset):.2f}%')

    # over_path = base_path
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        model.not_extra()
        model.reset()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        # additive_act, eigenvalues_count = whole_data_analysis_forward_pass(model, 'test', base_path, test_x)

    test_loss /= len(test_loader.dataset)
    print(f'test loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
    print("Training Complete")

    return train_add, train_eig, val_add, val_eig