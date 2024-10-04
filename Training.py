from Models import *
from utils import *
from DataGenerator import *
from ForwardPass import *
from Analysis import *
import gc
import wandb
import psutil
import torch.optim as optim


def train_model(model, train_loader, test_loader, base_path, train_x, train_y, val_x, val_y, test_x, test_y, val_loader=None, epochs=100, learning_rate=0.001, loss='crossentropy', optimizer='adam', analye_b_size=10000, status='default'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # type: ignore
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=5e-5)     # type: ignore
    
    if loss == 'nill':
        criterion = nn.NLLLoss()
    elif loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'mse':
        criterion = F.mse_loss()

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6, last_epoch=-1, verbose=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    print("start training")

    # Create or clear the result file
    result_file_path = base_path + "res.txt"
    with open(result_file_path, "w") as f:
        f.write("Training and Validation Results\n")

    for epoch in tqdm(range(epochs)):
        model.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            loss = loss.detach()
            optimizer.step()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        
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
        if (epoch + 1) % 10 == 0:
            # Create directory for the current set of 10 epochs
            save_dir = base_path + f"epoch_{epoch+1}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save the model
            torch.save(model.state_dict(), save_dir + "model.pth")
            print(f"Model saved at epoch {epoch + 1}")
            
            # Call the analysis function and save the results
            fixed_model_batch_analysis(model, train_x, train_y, device, '{}_{}'.format(save_dir, 'train_'),  '{}_'.format(status, str(epoch)), batch_size=analye_b_size)
            fixed_model_batch_analysis(model, val_x, val_y, device, '{}_{}'.format(save_dir, 'val_'), '{}_'.format(status, str(epoch)), batch_size=analye_b_size)
        
        gc.collect()
        # Empty the GPU cache
        torch.cuda.empty_cache()
        process = psutil.Process()
        memory_info = process.memory_info().rss / 1024 **2

        
        # Log GPU memory usage manually
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # Convert to MB
        wandb.log({
            "Train Loss": loss,
            "gpu_memory_allocated_MB": gpu_memory_allocated,
            "gpu_memory_reserved_MB": gpu_memory_reserved,
            "Val Loss": val_loss, 
            "Arrucary": accuracy,
            "CPU memory": memory_info
        })

        # Write results to the file
        with open(result_file_path, "a") as f:
            f.write(f'Epoch {epoch+1}/{epochs}, Val loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%\n')

        print(f'Epoch {epoch+1}/{epochs}, Val loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%, Learning Rate: {current_lr:.6f}')

    
    print("Training Complete")

    fixed_model_batch_analysis(model, test_x, test_y, device, '{}_{}'.format(base_path, 'test_'), status, batch_size=analye_b_size)

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
