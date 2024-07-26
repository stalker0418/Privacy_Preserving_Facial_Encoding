import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import time
import os
from tensorflow import keras
from keras import layers
import copy
import cv2
import pandas as pd
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau



epsilon = 8.0  
sensitivity = 1.0 


encoding_techniques = ['original', 'single_conv', 'double_conv', 'patched', 'psuedo_dp', 'pseudo_dp_patched', 'dyn_dp', 'dyn_dp_patched', 'single_conv_dyn_dp' ]

def normalize_dataset(dataset):
    # Assuming 'dataset' is a 3D NumPy array representing images
    # Normalize to the range [0, 1]
    normalized_dataset = (dataset.astype(np.float32) / 255.0).clip(0.0, 1.0)

    # Alternatively, normalize to the range [-1, 1]
    # normalized_dataset = (dataset.astype(np.float32) / 127.5) - 1.0

    return normalized_dataset



def extract_numerical_value(file_name):
    return int(file_name.split('_')[2])

# Read the images, and update the y_label so that missing images are removed from the labels
def read_images(image_folder, target_length):
    image_files = os.listdir(image_folder)

    image_files = sorted(image_files, key=extract_numerical_value)

    # Initialize an empty list to store images
    x_train, y_train, label_count = [], [], [0] * (target_length)

    # Loop through each image file, read it as grayscale, and append to x_train
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = int(image_file.split('_')[4].split('.')[0])

        # Check if the image is successfully loaded
        if img is not None: 
            # if label_count[label] <=cut_off_length:
            x_train.append(img)
            y_train.append(label)   
            label_count[label] += 1                                                                 
        else:
            print(f"Failed to load {image_file}")

    # Convert the list of images to a NumPy array
    x_train, y_train = np.array(x_train), np.array(y_train)

    # expect_num = set(range(0,len(y_train)))


    # existing_num = {int(file.split('_')[2]) for file in image_files}

    # missing_num = expect_num - existing_num
    # y_train = np.delete(y_train, list(missing_num))

    print("label count for {image_folder} is :", label_count)
    return x_train, y_train


def read_images_for_mask(image_folder):

    image_files = os.listdir(image_folder)

    image_files = sorted(image_files)

    # Initialize an empty list to store images
    x_train, y_train = [], []

    # Loop through each image file, read it as grayscale, and append to x_train
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image is successfully loaded
        if img is not None:
            x_train.append(img)
            # y_train.append(int(image_file.split('_')[4]))
            y_train.append(0) if (len(image_file.split('_')) == 3) else y_train.append(1)
            # if len(image_file.split('_')!=3): print(image_file)
        else:
            print(f"Failed to load {image_file}")

    # Convert the list of images to a NumPy array
    x_train, y_train = np.array(x_train), np.array(y_train)


    return x_train, y_train


# Generate Laplace noise based on privacy parameters.
def laplace_mechanism(data, epsilon, sensitivity):
    np.random.seed(42)
    scale = sensitivity / epsilon
    # noise = np.random.laplace(0.0, scale, data.shape)
    noise = np.random.laplace(0.0, scale, data.shape).astype(np.float32)
    noisy_image =  data + noise
    # print(noise)
    return noisy_image

def laplace_mechanism_dynamic(data, epsilon):
    new_data = copy.deepcopy(data)
    for i in range(1,len(data)):
        prev, pres = data[i-1], data[i]
        prev_set, pres_set = set(prev), set(pres)
        # jaccard_similarity = len(prev_set.intersection(pres_set)) / len(prev_set.union(pres_set))
        sensitivity = (np.linalg.norm(pres - prev))
        # sensitivity = (np.linalg.norm(pres - prev))/(jaccard_similarity + 0.8)
        scale = sensitivity/epsilon
        noise = np.random.laplace(0.0,scale,data[i-1].shape).astype(np.float32)
        # print(sensitivity, noise.shape)
        new_data[i-1] = data[i-1] + noise
    return new_data



def train_model(model, train_loader, val_loader, criterion, optimizer,  num_epochs=25, is_train=True):
    since = time.time()
    
    acc_history = []
    loss_history = []

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        print('Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_accuracy))

        # Validation
        model.eval()  # Set the model to evaluation mode
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0
        

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            val_loss = val_running_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

        # scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history   


def main():
    lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.8)


    X = lfw_people.images
    Y = lfw_people.target
    target_names = lfw_people.target_names

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # print(x_train.shape, y_train.shape, x_test.shape, type(x_train[0]))
    # y_train_indices_to_delete = [896,706,133,1030,39,10,911,496,19,918,889,443]
    # y_test_indices_to_delete = [64,205,23,57,31]


    train_folder = "C:/Users/pakal/Downloads/privacy_encoding/lfw_people_images_40/x_train/"
    test_folder  = "C:/Users/pakal/Downloads/privacy_encoding/lfw_people_images_40/x_test/"

    train_folder_mask_label = "/data/manassanjay/privacy_encoding/lfw_people_images_40_masked_gray/x_train"
    test_folder_mask_label = "/data/manassanjay/privacy_encoding/lfw_people_images_40_masked_gray/x_test"

    print("Test message to see that code reached here")

    (x_train,y_train), (x_test,y_test) = read_images(train_folder, len(target_names)), read_images(test_folder, len(target_names))


    # (x_train,y_train), (x_test,y_test) = read_images_for_mask(train_folder_mask_label), read_images_for_mask(test_folder_mask_label)
    # Shuffle training data
    # x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Shuffle testing data
    # x_test, y_test = shuffle(x_test, y_test, random_state=42)
    
    x_train, x_test = normalize_dataset(x_train), normalize_dataset(x_test)
    # x_train, x_test = np.array(x_train), np.array(x_test)
    
    # y_train,y_test = np.delete(y_train,y_train_indices_to_delete), np.delete(y_test, y_test_indices_to_delete)

    print("The new masked shapes are:", x_train.shape,y_train.shape, x_test.shape, y_test.shape)

    # for i in y_train: print(i)
    # print("----------------------------------------------------------------------------------------------------")

    # for i in y_test: print(i)
    



    # Model used to perform Patching on Images
    patching_model = keras.Sequential([
        layers.Conv2D(filters=32,kernel_size= (5,5), strides= (5,5), activation= 'relu', input_shape = (x_train.shape[1], x_test.shape[2], 1)),
        layers.MaxPooling2D(pool_size=(2,2))
    ])

    encode_model = keras.models.Model(inputs = patching_model.inputs, outputs = patching_model.layers[0].output)
    print(encode_model.summary())

    # Model used for feature extraction for Single Conv and Double Conv
    conv_model = keras.Sequential([
        layers.Conv2D(filters=32,kernel_size= (7,7), strides=(1,1), activation= 'relu', input_shape = (x_train.shape[1], x_train.shape[2], 1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=64,kernel_size= (5,5), activation= 'relu', input_shape = (x_train.shape[1], x_train.shape[2], 1))
    ])

    conv_encode_model_2 = keras.models.Model(inputs = conv_model.inputs, outputs = conv_model.layers[2].output)
    conv_encode_model_1 = keras.models.Model(inputs = conv_model.inputs, outputs = conv_model.layers[0].output)

    print(conv_encode_model_2.summary(), conv_encode_model_1.summary())


    x_patched_train, x_patched_test = encode_model(x_train), encode_model(x_test)
    x_conv_train_1, x_conv_test_1 = conv_encode_model_1(x_train), conv_encode_model_1(x_test)
    x_conv_train_2, x_conv_test_2 = conv_encode_model_2(x_train), conv_encode_model_2(x_test)
    x_patched_train, x_patched_test = np.sum(x_patched_train, axis=3), np.sum(x_patched_test, axis = 3)
    x_conv_train_1, x_conv_test_1 = np.sum(x_conv_train_1, axis = 3), np.sum(x_conv_test_1, axis=3)
    x_conv_train_2, x_conv_test_2 = np.sum(x_conv_train_2, axis = 3), np.sum(x_conv_test_2, axis=3)

    x_train_only_dp, x_test_only_dp = [], []
    x_dp_train, x_dp_test = [], []
    x_dyn_dp_train, x_dyn_dp_test = [], []
    x_conv_dp_train, x_conv_dp_test = [], []
    x_patched_dp_dyn_train, x_patched_dp_dyn_test = [], []

    for i in range(x_train.shape[0]):
        x_train_only_dp.append(laplace_mechanism(x_train[i],8,1))
    for i in range(x_test.shape[0]):
        x_test_only_dp.append(laplace_mechanism(x_test[i],8,1))

    for i in range(x_patched_train.shape[0]):
        x_dp_train.append(laplace_mechanism(x_patched_train[i],8,1))

    for i in range(x_patched_test.shape[0]):
        x_dp_test.append(laplace_mechanism(x_patched_test[i],8,1))

    for i in range(x_conv_train_1.shape[0]):
        x_conv_dp_train.append(laplace_mechanism_dynamic(x_conv_train_1[i],8))

    for i in range(x_conv_test_1.shape[0]):
        x_conv_dp_test.append(laplace_mechanism_dynamic(x_conv_test_1[i],8))

    for i in range(x_patched_train.shape[0]):
        x_patched_dp_dyn_train.append(laplace_mechanism_dynamic(x_patched_train[i],8))

    for i in range(x_patched_test.shape[0]):
        x_patched_dp_dyn_test.append(laplace_mechanism_dynamic(x_patched_test[i],8))


    for i in range(x_train.shape[0]):
        x_dyn_dp_train.append(laplace_mechanism_dynamic(x_train[i],8))

    for i in range(x_test.shape[0]):
        x_dyn_dp_test.append(laplace_mechanism_dynamic(x_test[i],8))

    print(x_dp_train[0].shape, x_dp_test[0].shape, x_conv_dp_train[0].shape, x_patched_dp_dyn_train[0].shape)







    # 


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    aug_transform_90 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(90,90)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    aug_transform_180 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(180,180)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    aug_transform_270 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(270,270)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    aug_transform_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    aug_transform_flip_random = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomRotation(degrees=(-20,20)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    aug_transform_random = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(-20,20)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # 

    encoded_dataset = [(x_train,x_test), (x_conv_train_1, x_conv_test_1), (x_conv_train_2, x_conv_test_2), (x_patched_train, x_patched_test), (x_train_only_dp,x_test_only_dp), (x_dp_train, x_dp_test), (x_dyn_dp_train, x_dyn_dp_test),(x_patched_dp_dyn_train, x_patched_dp_dyn_test), (x_conv_dp_train, x_conv_dp_test) ]
    # encoded_dataset = [(x_train, x_test)]
    # Pre-Trained Resnet-18 used as a standard recognition model to check our encodings
    resnet18 = models.resnet18(pretrained=True)

    #It is intially taking 3 channels resnet, so we convert the first layer to take only one channel here
    resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    #Part of transfer learning to not update the feature learning weights
    for p in resnet18.parameters():
        p.requires_grad = False

    unfreeze_layers = ['layer4']

    for name, param in resnet18.named_parameters():
        if any(layer_name in name for layer_name in unfreeze_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False


    resnet18.fc = nn.Linear(512,len(target_names))


    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr = 0.004,momentum=.9)
    
    count = 0
    for i in encoded_dataset:
        
        # Define a learning rate scheduler
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

        
        x_tensor_train, x_tensor_test = torch.stack([transform(image) for image in i[0]]), torch.stack([transform(image) for image in i[1]])
        y_tensor_train, y_tensor_test = torch.LongTensor(y_train), torch.LongTensor(y_test)
        # print(x_tensor_train.shape, y_tensor_train.shape, x_tensor_test.shape, y_tensor_test.shape)

        original_dataset_train, original_dataset_test = TensorDataset(x_tensor_train, y_tensor_train), TensorDataset(x_tensor_test, y_tensor_test)
        augmented_datasets_train, augumented_datasets_test = [original_dataset_train], [original_dataset_test]

        # Number of times to augment the dataset
        augmentation_factor = 4

        # # Apply augmentation to create additional datasets
        # for _ in range(augmentation_factor):
        #     augmented_dataset_train, augumented_dataset_test = TensorDataset(torch.stack([aug_transform(image) for image in i[0]]), y_tensor_train), TensorDataset(torch.stack([aug_transform(image) for image in i[1]]), y_tensor_test)
        #     augmented_datasets_train.append(augmented_dataset_train)
        #     augumented_datasets_test.append(augumented_dataset_test)

        
        # augmented_dataset_train, augumented_dataset_test = TensorDataset(torch.stack([aug_transform_flip(image) for image in i[0]]), y_tensor_train), TensorDataset(torch.stack([aug_transform_flip(image) for image in i[1]]), y_tensor_test)
        # augmented_datasets_train.append(augmented_dataset_train)
        # augumented_datasets_test.append(augumented_dataset_test)

        augmented_dataset_train, augumented_dataset_test = TensorDataset(torch.stack([aug_transform_random(image) for image in i[0]]), y_tensor_train), TensorDataset(torch.stack([aug_transform_random(image) for image in i[1]]), y_tensor_test)
        augmented_datasets_train.append(augmented_dataset_train)
        augumented_datasets_test.append(augumented_dataset_test)

        augmented_dataset_train, augumented_dataset_test = TensorDataset(torch.stack([aug_transform_flip_random(image) for image in i[0]]), y_tensor_train), TensorDataset(torch.stack([aug_transform_flip_random(image) for image in i[1]]), y_tensor_test)
        augmented_datasets_train.append(augmented_dataset_train)
        augumented_datasets_test.append(augumented_dataset_test)

        augmented_dataset_train, augumented_dataset_test = TensorDataset(torch.stack([aug_transform_flip(image) for image in i[0]]), y_tensor_train), TensorDataset(torch.stack([aug_transform_flip(image) for image in i[1]]), y_tensor_test)
        augmented_datasets_train.append(augmented_dataset_train)
        augumented_datasets_test.append(augumented_dataset_test)

        # Concatenate the datasets to create the final training dataset
        final_dataset_train, final_dataset_test = torch.utils.data.ConcatDataset(augmented_datasets_train), torch.utils.data.ConcatDataset(augumented_datasets_test)


        print(len(final_dataset_train), len(final_dataset_test))

        print("Finished converting to tensors")

        # plt.subplot(1,1,1)
        # plt.grid('off')
        # plt.axis("off")
        # plt.imshow(x_tensor_train[1].cpu().numpy()[0], cmap='gray')
        # plt.savefig('image_output/'+encoding_techniques[count] +'.png')


        print("Finished saving the sample of the image used for train")

        train_loader = DataLoader(TensorDataset(x_tensor_train, y_tensor_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_tensor_test, y_tensor_test), batch_size=32)

        # train_loader = DataLoader(final_dataset_train, batch_size=32, shuffle=True)
        # test_loader = DataLoader(final_dataset_test, batch_size=32, shuffle=True)


        

        train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train_model(resnet18, train_loader, test_loader,  lossfun, optimizer, num_epochs= 25)

        
        epochs_range = range(1,26)
        # Create a DataFrame for your list values
        data = {
            'Epochs': list(epochs_range),
            'Training Loss': train_loss_hist,
            'Validation Loss': val_loss_hist,
            'Training Accuracy': train_acc_hist,
            'Validation Accuracy': val_acc_hist
        }

        df = pd.DataFrame(data)

        excel_dir = 'metrics/mod_vs_mod/gaivi/layer4_unfreezed/min_40_no_aug_latest/'

        if not os.path.exists(excel_dir): os.makedirs(excel_dir)

        # Define the file path for the Excel file
        excel_file_path = excel_dir+encoding_techniques[count] +'.xlsx'

        # Save the DataFrame to an Excel file
        df.to_excel(excel_file_path, index=False)
        print(f'----------------------------------Finished executing the method {encoding_techniques[count]}-----------------------------------------------------')
        count += 1
 

if __name__ == "__main__":
    main()