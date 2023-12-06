from azure.storage.blob import BlobServiceClient, ContainerClient, BlobPrefix
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import time
import cv2
import imageio.v3 as iio
import os
import matplotlib.pyplot as plt
from copy import deepcopy

# Get the directory of the current script file
os.chdir(r'C:\Users\Ecko_\exjobb_jupyter')
#print(os.getcwd())

class BlobSamples(object):
    def __init__(self):
        self.depth = 0
        self.indent = "  "

    def list_blobs_hierarchical(self, container_client: ContainerClient, prefix):
        for blob in container_client_instance.walk_blobs(name_starts_with=prefix, delimiter='/'):
            if isinstance(blob, BlobPrefix):
                # Indentation is only added to show nesting in the output
                print(f"{self.indent * self.depth}{blob.name}")
                self.depth += 1
                self.list_blobs_hierarchical(container_client_instance, prefix=blob.name)    
                self.depth -= 1
                #print('hi')
            else:
                print(f"{self.indent * self.depth}{blob.name}")

    def load_videos_into_memory(self, blob_service_client: BlobServiceClient, container_name, folder_name, videos_loaded):
        container_client = blob_service_client.get_container_client(container=container_name)

        blob_list = container_client.list_blobs(name_starts_with=folder_name)

        videos = []
        counter = 0  # Add a counter
        for blob in blob_list:
            if counter >= videos_loaded:  # Break the loop after loading 5 videos
                break
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            video_data = blob_client.download_blob().readall()

            # Create a dictionary for the video
            video = {
                'path': blob.name,  # The path of the video
                'data': video_data  # The bytes data of the video
            }

            videos.append(video)
            counter += 1  # Increment the counter
        return videos

class PreparedDataset(Dataset):
    def __init__(self, videos, trainval='train'):
        self.video_names_train = []
        self.video_names_test = []
        self.action_labels_train = []
        self.action_labels_test = []
        self.predata_train = []
        self.predata_test = []
        self.videos = videos
        self.trainval = trainval

        # Read the classes file
        with open('classInd.txt', 'r') as f:
            classes = f.readlines()
        classes = [c.strip().split(' ', 1)[1] for c in classes]
        self.class_to_id = {c: i for i, c in enumerate(classes)} #dictionary

        # First, read the paths and labels from the trainlist and testlist files into dictionaries
        train_paths = {}
        test_paths = {}
        with open('trainlist1.txt', 'r') as f:
            for line in f:
                path, label = line.strip().split(' ')
                train_paths[path] = int(label)-1  # Subtract 1 to make the labels 0-indexed
        with open('testlist1.txt', 'r') as f:
            test_paths = {line.strip() for line in f}
        for video in videos:
            path = video['path'][len('UCF-101/'):]  # Remove the 'UCF-101/' prefix from the video path
            video_name = path.split('/')[1].split('.avi')[0].replace('v_', '') #Extracting the name
            if path in train_paths:
                self.video_names_train.append(video_name)
                label = train_paths[path]  # Extract the label from the dictionary
                self.action_labels_train.append(label)
                self.predata_train.append(video)
            elif path in test_paths:
                self.video_names_test.append(video_name)
                class_name = path.split('/')[0]
                label = self.class_to_id[class_name]
                self.action_labels_test.append(label)
                self.predata_test.append(video)

    def __getitem__(self, index):  # https://stackoverflow.com/questions/43627405/understanding-getitem-method-in-python
        if self.trainval == 'train':
            video_name = self.video_names_train[index]
            label = self.action_labels_train[index]
            video = self.predata_train[index]
        else:
            video_name = self.video_names_test[index]
            label = self.action_labels_test[index]
            video = self.predata_test[index]
        return video, label, video_name

    def __len__(self):
        if self.trainval == "train":
            return len(self.predata_train)
        else:
            return len(self.predata_test)

# Print the first 5 file paths from filelist_train and filelist_test
def test_video_dataset(dataset, n):
    print(f"Number of training videos: {len(dataset.predata_train)}")
    print(f"Number of testing videos: {len(dataset.predata_test)}")

    fig, axs = plt.subplots(nrows=(n+2)//3, ncols=3, figsize=(15, 5*(n+2)//3))

    # Display the first n videos from the training data
    for i, video_data in enumerate(dataset.predata_train[:n]):  # Only take the first n videos from training data
        frames = iio.imread(video_data['data'], index=None, format_hint=".avi")
        print(frames.shape)
        frame = frames[0]

        # Display the frame using pyplot
        axs[i//3, i%3].imshow(frame)
        axs[i//3, i%3].set_title(f'Frame from video {dataset.video_names_train[i]}')
        axs[i//3, i%3].axis('off')

    plt.tight_layout()
    plt.show()

class PreprocessedTemporalFourData(Dataset):
    def __init__(self, dataset, trainval='train', mean=[96.5, 96.5, 96.5], imagesize=80):
        self.dataset = dataset
        self.trainval = trainval
        self.mean = mean
        self.imagesize = imagesize
        
    def __len__(self):
        if self.trainval == 'train':
            return len(self.dataset.predata_train)
        else:
            return len(self.dataset.predata_test)

    def __getitem__(self, index):
        video, action_label, video_name = self.get_video_data(index)
        # Read the frames from the video
        frames = iio.imread(video['data'], index=None, format_hint=".avi")

        # Drop every other frame
        frames = frames[::2]
        
        # Compute the optical flow weights for frame selection
        #start_time = time.time()
        weights, flows = self.compute_optical_flow_weights(frames)
        #end_time = time.time()
        #print(f'Time taken for retrieving flows: {end_time - start_time} seconds')
                
        # Select the frames from the video based on the weights of the optical flow
        indices = np.random.choice(frames.shape[0], size=4, replace=False, p=weights)
        selected_frames = frames[indices]

        # Normalize the indices to their respective order
        order_indices = indices.argsort().argsort()

        # Create label list for the frame order
        frame_order_label, frames_canonical_order = self.get_frame_order_label(order_indices)
        
        # Preprocess the frames
        preprocessed_frames, uncropped_frames = self.preprocess_frames(selected_frames)        

        # Convert the action_label, flows and indices to PyTorch tensors
        action_label = torch.tensor(action_label)
        flows = torch.tensor(np.array(flows))
        indices = torch.tensor(indices)

        return preprocessed_frames, frame_order_label, action_label, video_name, frames_canonical_order, uncropped_frames

    def get_video_data(self, index):
        if self.trainval == 'train':
            video = self.dataset.predata_train[index]
            action_label = self.dataset.action_labels_train[index]
            video_name = self.dataset.video_names_train[index]
        else:
            video = self.dataset.predata_test[index]
            action_label = self.dataset.action_labels_test[index]
            video_name = self.dataset.video_names_test[index]
        return video, action_label, video_name
    
    def compute_optical_flow_weights(self, frames):
        # Downsample the frames to optimize the optical flow computation # TODO figure out the optimal downsampling size
        downsampled_frames = [cv2.resize(frame, (160, 80)) for frame in frames]

        # Compute the optical flow between frames
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in downsampled_frames]
        flows = [cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(gray_frames)-1)]
        # Compute the magnitude of the optical flow
        magnitudes = [np.sqrt(flow[...,0]**2 + flow[...,1]**2) for flow in flows]
        # Compute the average flow magnitude per frame
        avg_magnitudes = [np.mean(magnitude) for magnitude in magnitudes]
        # Append a zero to avg_magnitudes
        avg_magnitudes.append(0)
        # Use the average flow magnitude as a weight for frame selection
        weights = avg_magnitudes / np.sum(avg_magnitudes)
        return weights, flows
    
    def get_frame_order_label(self, order_indices):
        frame_order_to_label_dict = {
            (0, 1, 2, 3): 0,
            (0, 2, 1, 3): 1,
            (0, 3, 2, 1): 2,
            (0, 1, 3, 2): 3,
            (0, 3, 1, 2): 4,
            (0, 2, 3, 1): 5,
            (1, 0, 2, 3): 6,
            (1, 0, 3, 2): 7,
            (1, 2, 0, 3): 8,
            (1, 3, 0, 2): 9,
            (2, 0, 1, 3): 10,
            (2, 1, 0, 3): 11
        }
        frames_canonical_order = order_indices if order_indices[0] < order_indices[-1] else order_indices[::-1]
        frame_order_label = frame_order_to_label_dict[tuple(frames_canonical_order)]
        return torch.tensor(frame_order_label), torch.tensor(frames_canonical_order.copy())

    def preprocess_frames(self, selected_frames):
        # Define the jittering outside the loop #TODO this is also where the patch selection should be done
        sjdis = 5
        startx = np.random.randint(0, selected_frames[0].shape[1] - self.imagesize)
        starty = np.random.randint(0, selected_frames[0].shape[0] - self.imagesize)
        shift_x = np.random.randint(-sjdis, sjdis)
        shift_y = np.random.randint(-sjdis, sjdis)
        preprocessed_frames = []
        uncropped_frames = []
        for i, frame in enumerate(selected_frames):
            frame = Image.fromarray(frame.astype('uint8'))
            frame = np.array(frame)
            #TODO # Subtract the mean: #frame = frame - self.mean
            #TODO # Applying mirroring
            #TODO # Apply channel splitting
            preprocessed_frame = self.spatial_jitter(frame, startx, starty, shift_x, shift_y)
            uncropped_frames.append(torch.from_numpy(frame.copy()))
            preprocessed_frames.append(torch.from_numpy(preprocessed_frame.copy()))
        return torch.stack(preprocessed_frames), torch.stack(uncropped_frames)

    def spatial_jitter(self, frame, startx, starty, shift_x, shift_y):
        # Define the size of the crop
        sjx = self.imagesize
        sjy = self.imagesize
        # Define the end coordinates of the crop    
        endx = startx + sjx
        endy = starty + sjy
        if startx + shift_x > 0 and endx + shift_x < frame.shape[1]:
            newx = startx + shift_x
        else:
            newx = startx
        if starty + shift_y > 0 and endy + shift_y < frame.shape[0]:
            newy = starty + shift_y
        else:
            newy = starty
        # Crop the image
        frame = frame[newy:newy+sjy, newx:newx+sjx]
        return frame

    def channel_split(self, frame):
        # This is a simplified version of the channel splitting code in the provided function
        rgb = random.randint(0, 2)
        frame = np.array(frame)[:,:,rgb]
        frame = np.stack((frame,)*3, axis=2)
        return frame

def test_temporal_four(temporal_four, n):
    input_frames, frame_order_label, action_label, video_name, frames_canonical_order, uncropped_frames = temporal_four[2]
    
    # Reorder the uncropped_frames based on frames_canonical_order
    ordered_uncropped_frames = uncropped_frames[frames_canonical_order]

    visualize_frames(input_frames, uncropped_frames, ordered_uncropped_frames, frames_canonical_order, video_name)
    #visualize_optical_flow(flows, indices) # TODO add method to get_flows&indices

def visualize_frames(input_frames, uncropped_frames, ordered_uncropped_frames, frames_canonical_order, video_name):
    # Create a figure and a grid of subplots with an extra row for the column titles
    fig, axs = plt.subplots(4, 3, figsize=(10, 10))

    for i, (frame, uncropped_frame, ordered_frame) in enumerate(zip(input_frames, uncropped_frames, ordered_uncropped_frames)):
        frame = frame.numpy()
        uncropped_frame = uncropped_frame.numpy()
        ordered_frame = ordered_frame.numpy()

        axs[i, 0].imshow(ordered_frame)
        axs[i, 0].set_title(f'Frame {i}', fontsize=8)
        axs[i, 0].axis('off')

        axs[i, 1].imshow(uncropped_frame)
        axs[i, 1].set_title(f'Frame {frames_canonical_order[i]}', fontsize=8)
        axs[i, 1].axis('off')

        axs[i, 2].imshow(frame)
        axs[i, 2].set_title(f'Frame {frames_canonical_order[i]}', fontsize=8)
        axs[i, 2].axis('off')

    # Set a common title for the entire plot
    fig.suptitle(f'Frame Analysis of video {video_name}', fontsize=16)

    # Define the titles for each column
    column_titles = ['Uncropped Frames', 'Shuffled Uncropped Frames', 'Shuffled Preprocessed Frames']

    # Set the column titles manually using the text function
    fig.text(0.25, 0.93, column_titles[0], ha='center', va='center', fontsize=10)
    fig.text(0.5, 0.93, column_titles[1], ha='center', va='center', fontsize=10)
    fig.text(0.8, 0.93, column_titles[2], ha='center', va='center', fontsize=10)

    plt.show()

def visualize_optical_flow(flows, indices):
    # Reorder the flows based on indices
    flows = [flows[i] for i in indices[:-1]]

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 

    # Visualize the optical flow
    for i in range(3):
        flow = flows[i]
        flow = flow.numpy()
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[...,1] = 255

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        # New code
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mag = np.uint8(mag)
        hsv[...,2] = mag
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        im = axs[i].imshow(rgb)
        axs[i].set_title(f'Optical flow from frame {i} to its next', fontsize=10)
        axs[i].axis('off')

        # Add a color bar
        fig.colorbar(im, ax=axs[i], orientation='vertical')
        
    # Display the figure with the subplots
    plt.tight_layout()
    plt.show()