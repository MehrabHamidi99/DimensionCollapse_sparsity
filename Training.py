from Models import *
from utils import *
from DataGenerator import *
from ForwardPass import *
from Analysis import *


def train_model(model, train_loader, test_loader, base_path, train_x, train_y, val_x, val_y, test_x, test_y, val_loader=None, epochs=100, learning_rate=0.001, loss='nill'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    if loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    if loss == 'mse':
        criterion = F.mse_loss()

    print("start training")

    # Create or clear the result file
    result_file_path = base_path + "res.txt"
    with open(result_file_path, "w") as f:
        f.write("Training and Validation Results\n")

    for epoch in tqdm(range(epochs)):
        model.train()
        # over_path = base_path + 'epoch_{}/'.format(str(epoch))
        # if not os.path.isdir(over_path):
        #     os.makedirs(over_path)
        
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = torch.argmax(torch.exp(output), dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)

        # Save model every 10 epochs
        if (epoch + 1) % 1 == 0:
            # Create directory for the current set of 10 epochs
            save_dir = base_path + f"epoch_{epoch+1}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save the model
            torch.save(model.state_dict(), save_dir + "model.pth")
            print(f"Model saved at epoch {epoch + 1}")
            
            # Call the analysis function and save the results
            fixed_model_batch_analysis(model, train_x, train_y, device, '{}_{}'.format(save_dir, 'train_'), '784, [256, 128, 64, 32, 10]')
            fixed_model_batch_analysis(model, val_x, val_y, device, '{}_{}'.format(save_dir, 'val_'), '784, [256, 128, 64, 32, 10]')

        # Write results to the file
        with open(result_file_path, "a") as f:
            f.write(f'Epoch {epoch+1}/{epochs}, Val loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%\n')

        print(f'Epoch {epoch+1}/{epochs}, Val loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    
    print("Training Complete")

    fixed_model_batch_analysis(model, test_x, test_y, device, '{}_{}'.format(base_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    # Final stats calculation
    def print_stats(loader, mode):
        loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss += criterion(output, target).item()
                pred = torch.exp(output).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        with open(result_file_path, "a") as f:
            f.write(f'{mode} loss: {loss:.4f}, Accuracy: {accuracy:.2f}%\n')

    # Write the final stats for train, validation, and test sets
    print_stats(train_loader, 'train')
    print_stats(val_loader, 'val')
    print_stats(test_loader, 'test')
