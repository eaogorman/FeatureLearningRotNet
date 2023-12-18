import os
import re
import matplotlib.pyplot as plt

def extract_model_stats(model_paths):
    all_epoch_data = []

    for model_path in model_paths:
        # Open the file and read its contents
        with open(model_path, 'r') as file:
            log_content = file.readlines()

        # Initialize variables for the current model
        epoch_data = []
        current_epoch = None
        current_training_stats = None
        current_evaluation_stats = None

        # Iterate over each line in the log
        for line in log_content:
            # Check if the line contains information about the current epoch
            if "Training epoch" in line:
                # If there is data from the previous epoch, store it
                if current_epoch is not None:
                    epoch_data.append({
                        "epoch": current_epoch,
                        "training_stats": current_training_stats,
                        "evaluation_stats": current_evaluation_stats
                    })

                # Extract the epoch number
                current_epoch = int(re.search(r"Training epoch \[\s*(\d+)", line).group(1))
                # Reset the current training and evaluation stats
                current_training_stats = None
                current_evaluation_stats = None

            # Check if the line contains information about training stats
            elif "==> Training stats:" in line:
                # Extract the dictionary from the line
                match = re.search(r"==> Training stats: (.+)", line)
                if match:
                    current_training_stats = eval(match.group(1))

            # Check if the line contains information about evaluation stats
            elif "==> Evaluation stats:" in line:
                # Extract the dictionary from the line
                match = re.search(r"==> Evaluation stats: (.+)", line)
                if match:
                    current_evaluation_stats = eval(match.group(1))

        # If there is data from the last epoch, store it
        if current_epoch is not None:
            epoch_data.append({
                "epoch": current_epoch,
                "training_stats": current_training_stats,
                "evaluation_stats": current_evaluation_stats
            })

        # Store the epoch data for the current model
        all_epoch_data.append(epoch_data)

    return all_epoch_data    

    # # Plotting
    # plt.figure(figsize=(15, 5))



    # # Plot Test Loss
    # plt.subplot(1, 3, 2)
    # for i, epoch_data in enumerate(all_epoch_data):
    #     epochs = [entry['epoch'] for entry in epoch_data]
    #     evaluation_loss = [entry['evaluation_stats']['loss'] for entry in epoch_data]
    #     plt.plot(epochs, evaluation_loss, label=f'Model {i+1}')

    # plt.title('Test Loss Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Plot Training Accuracy and Evaluation Accuracy
    # plt.subplot(1, 3, 3)
    # for i, epoch_data in enumerate(all_epoch_data):
    #     epochs = [entry['epoch'] for entry in epoch_data]
    #     training_accuracy = [entry['training_stats']['prec1'] for entry in epoch_data]
    #     evaluation_accuracy = [entry['evaluation_stats']['prec1'] for entry in epoch_data]
    #     plt.plot(epochs, training_accuracy, label=f'Training Model {i+1}', linestyle='--')
    #     plt.plot(epochs, evaluation_accuracy, label=f'Evaluation Model {i+1}', linestyle='-')

    # plt.title('Accuracy Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # Plot Training Loss
def plot_training_loss(all_epoch_data, models, colors):
    for i, epoch_data in zip(enumerate(all_epoch_data), models):
        epochs = [entry['epoch'] for entry in epoch_data]
        training_loss = [entry['evaluation_stats']['loss'] for entry in epoch_data]
        plt.plot(epochs, training_loss, label=models[i], c=colors[i])

def plot_test_loss(all_epoch_data, models, colors):
    for i, epoch_data in zip(enumerate(all_epoch_data), models):
        epochs = [entry['epoch'] for entry in epoch_data]
        evaluation_loss = [entry['evaluation_stats']['loss'] for entry in epoch_data]
        plt.plot(epochs, evaluation_loss, label=models[i], c=colors[i])

def plot_test_accuracy(all_epoch_data, models, colors):
    for i, epoch_data in zip(enumerate(all_epoch_data), models):
        epochs = [entry['epoch'] for entry in epoch_data]
        evaluation_accuracy = [entry['evaluation_stats']['prec1'] for entry in epoch_data]
        plt.plot(epochs, evaluation_accuracy, label=models[i], c=colors[i])

def plot_train_accuracy(all_epoch_data, models):
    for i, epoch_data in enumerate(all_epoch_data):
        epochs = [entry['epoch'] for entry in epoch_data]
        training_accuracy = [entry['training_stats']['prec1'] for entry in epoch_data]
        plt.plot(epochs, training_accuracy, label=models[i], linestyle='--')

# Example usage:
model_paths = [
    "/home/eao21/ece661/FeatureLearningRotNet/experiments/CIFAR10_LinearClassifier_on_RotNet_NIN4blocks_Conv2_feats_64_K2500/logs/LOG_INFO_2023-12-16_17:52:47.269520.txt",
    "/home/eao21/ece661/FeatureLearningRotNet/experiments/CIFAR10_LinearClassifier_on_RotNet_NIN4blocks_Conv2_feats_64_K1000/logs/LOG_INFO_2023-12-16_18:05:15.964140.txt"
]

extract_model_stats(model_paths)
