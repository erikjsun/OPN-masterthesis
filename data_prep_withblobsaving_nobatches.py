from azure.storage.blob import BlobServiceClient, ContainerClient, BlobPrefix
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torchviz import make_dot
from copy import deepcopy
import cv2
import imageio.v3 as iio
from PIL import Image
from torch.utils.data import Dataset
import io

# Azure Blob Storage configuration
STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
CONTAINERNAME = "exjobbssl"
FOLDERNAME = "UCF-101/HighJump/"  # Original data folder in blob storage
PREPROCESSEDDATA_FOLDERNAME = "preprocessed-data"  # Folder to save preprocessed data

# Initialize the BlobServiceClient
blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
container_client_instance = blob_service_client_instance.get_container_client(CONTAINERNAME)

class PreparedDataset(Dataset):
    def __init__(self, videos):
        self.video_names = []
        self.action_labels = []
        self.predata = []
        self.videos = videos
        self.create_video_data()

    def __getitem__(self, index):
        video_name = self.video_names[index]
        label = self.action_labels[index]
        video = self.predata[index]
        return video, label, video_name

    def __len__(self):
        return len(self.predata)

    def create_video_data(self):
        # Read the classes file
        with open('classInd.txt', 'r') as f:
            classes = f.readlines()
        classes = [c.strip().split(' ', 1)[1] for c in classes]
        self.class_to_id = {c: i for i, c in enumerate(classes)}  # Dictionary

        # Read trainlist1.txt into train_paths
        train_paths = {}
        with open('trainlist1.txt', 'r') as f:
            for line in f:
                path, label = line.strip().split(' ')
                train_paths[path] = int(label) - 1  # Subtract 1 to make the labels 0-indexed

        # Read testlist1.txt into test_paths
        test_paths = set()
        with open('testlist1.txt', 'r') as f:
            for line in f:
                test_paths.add(line.strip())

        # Process the videos
        for video in self.videos:
            path = video['path'][len('UCF-101/'):]  # Remove the 'UCF-101/' prefix from the video path
            video_name = path.split('/')[1].split('.avi')[0].replace('v_', '')  # Extracting the name

            # Get label
            if path in train_paths:
                label = train_paths[path]
            elif path in test_paths:
                class_name = path.split('/')[0]
                label = self.class_to_id[class_name]
            else:
                print(f"Video path '{path}' not found in train or test lists.")
                continue  # Skip this video

            self.video_names.append(video_name)
            self.action_labels.append(label)
            self.predata.append(video)

class PreprocessedTemporalFourData(Dataset):
    def __init__(self, dataset, pixel_mean=96.5, patch_size=160):
        self.dataset = dataset
        self.pixel_mean = pixel_mean
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset.predata)

    def __getitem__(self, index):
        video, action_label, video_name = self.get_video_data(index)
        return self.preprocess_video(video, action_label, video_name)

    def get_video_data(self, index):
        video = self.dataset.predata[index]
        action_label = self.dataset.action_labels[index]
        video_name = self.dataset.video_names[index]
        return video, action_label, video_name

    def preprocess_video(self, video, action_label, video_name):
        # Read the frames from the video
        frames = iio.imread(video['data'], index=None, format_hint=".avi")
        # Drop every other frame
        frames = frames[::2]

        # Compute the optical flow weights for frame selection
        weights, flows = self.compute_optical_flow_weights(frames)

        # Select the four frames from the video based on the weights of the optical flow – random order
        indices = np.random.choice(frames.shape[0], size=4, replace=False, p=weights)
        selected_frames = frames[indices]

        # Normalize the indices to their respective order
        ranked_indices = indices.argsort().argsort()

        # Saving the ordered frames for visualization
        ordered_indices = np.sort(indices)
        ordered_frames = frames[ordered_indices]

        # Create label list for the frame order
        frame_order_label, frames_canonical_order = self.get_frame_order_label(ranked_indices)

        # Applying random mirroring to all selected frames at once
        mirror = random.randint(0, 1)
        if mirror == 1:
            selected_frames_list = [np.array(Image.fromarray(frame.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT)) for frame in selected_frames]
            selected_frames = np.array(selected_frames_list)
            ordered_frames_list = [np.array(Image.fromarray(frame.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT)) for frame in ordered_frames]
            ordered_frames = np.array(ordered_frames_list)

        # Select the best patch from the frames
        best_patch = self.select_best_patch(selected_frames)

        # Preprocess the frames, including spatial jittering and channel splitting
        preprocessed_frames, preprocessed_frames_coordinates = self.preprocess_frames(selected_frames, best_patch)

        # Convert the action_label, flows and indices to PyTorch tensors
        action_label = torch.tensor(action_label)
        flows = torch.tensor(np.array(flows))
        indices = torch.tensor(indices)
        # print("Shape of preprocessed_frames:", preprocessed_frames.shape)
        return preprocessed_frames, frame_order_label, action_label, video_name, frames_canonical_order, selected_frames, preprocessed_frames_coordinates, ordered_frames

    def compute_optical_flow_weights(self, frames):
        # Downsample the frames to reduce the optical flow computation
        downsampled_frames = [cv2.resize(frame, (160, 80)) for frame in frames]

        # Compute the optical flow between frames
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in downsampled_frames]
        flows = [cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(gray_frames) - 1)]
        # Compute the magnitude of the optical flow
        magnitudes = [np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) for flow in flows]
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

    def select_best_patch(self, selected_frames):
        # Compute the optical flow between selected_frames
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in selected_frames]
        flows = [cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(gray_frames) - 1)]

        # Compute the magnitude of the optical flow
        magnitudes = [np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) for flow in flows]

        # Compute the sum of optical flow magnitudes for the selected frames
        summed_flow_magnitude = sum(magnitudes)

        # Initialize the best patch and its motion sum
        best_patch = None
        best_motion_sum = -1

        # Define the margin from the frame's edge where patches should not be selected
        margin = 30

        # Slide the patch over the summed frame, and find the patch with the largest motion sum
        for i in range(margin, summed_flow_magnitude.shape[0] - self.patch_size - margin + 1):
            for j in range(margin, summed_flow_magnitude.shape[1] - self.patch_size - margin + 1):
                # Compute the motion sum for the current patch
                current_motion_sum = summed_flow_magnitude[i:i + self.patch_size, j:j + self.patch_size].sum()

                # If the current motion sum is larger than the best one, update the best patch
                if current_motion_sum > best_motion_sum:
                    best_patch = (i, j)
                    best_motion_sum = current_motion_sum

        return best_patch

    def preprocess_frames(self, selected_frames, best_patch):
        # Define the spatial jittering distance
        sjdis = 20
        startx, starty = best_patch

        preprocessed_frames = []
        preprocessed_frames_coordinates = []
        for i, frame in enumerate(selected_frames):
            # Apply spatial jittering
            sj_frame, sj_frame_coordinates = self.spatial_jitter(frame, startx, starty, sjdis)

            preprocessed_frame = sj_frame
            preprocessed_frames.append(torch.from_numpy(preprocessed_frame.copy()))

            # Save the coordinates of the preprocessed frame
            preprocessed_frames_coordinates.append(sj_frame_coordinates)

        return torch.stack(preprocessed_frames), preprocessed_frames_coordinates

    def spatial_jitter(self, frame, startx, starty, sjdis):
        # Define the shift in pixels
        shift_x = np.random.randint(-sjdis, sjdis)
        shift_y = np.random.randint(-sjdis, sjdis)

        # Define the start coordinates of the crop
        newx = startx + shift_x
        newy = starty + shift_y

        # Crop the image
        sj_frame = frame[newx:newx + self.patch_size, newy:newy + self.patch_size]

        # Return the spatially jittered frame and its starting coordinates
        return sj_frame, (newx, newy)

class BlobSamples(object):
    def __init__(self, single_folder_mode=False, specific_folder_name=""):
        self.depth = 0
        self.indent = "  "
        self.folders = []
        self.loaded_folders = []
        self.single_folder_mode = single_folder_mode
        self.specific_folder_name = specific_folder_name

    def list_blobs_hierarchical(self, container_client: ContainerClient, prefix="", depth=0):
        for blob in container_client.walk_blobs(name_starts_with=prefix, delimiter='/'):
            if isinstance(blob, BlobPrefix):
                if depth > 0:
                    self.folders.append(blob.name)
                self.list_blobs_hierarchical(container_client, blob.name, depth + 1)
        return self.folders

    def load_videos_into_memory(self, blob_service_client, container_name, videos_loaded, folder_limit):
        container_client = blob_service_client.get_container_client(container_name)
        videos = []
        folder_count = 0
        videos_counter = 0

        # Determine folders to load videos from
        if self.single_folder_mode:
            folder_names = [self.specific_folder_name]
        else:
            self.folders = []  # Clear previous entries if any
            folder_names = self.list_blobs_hierarchical(container_client)[:folder_limit]

        print(f"Folder names to process: {folder_names}")  # Debug print

        for folder_name in folder_names:
            if folder_count >= folder_limit:
                break
            blob_list = container_client.list_blobs(name_starts_with=folder_name)
            print(f"Processing folder: {folder_name}")  # Debug print

            for blob in blob_list:
                if videos_loaded is not None and videos_counter >= videos_loaded:
                    break
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                video_data = blob_client.download_blob().readall()
                video = {'path': blob.name, 'data': video_data}
                videos.append(video)
                videos_counter += 1

            self.loaded_folders.append(folder_name)
            folder_count += 1

        return videos

    def get_loaded_folders(self):
        return self.loaded_folders

    def load_videos_generator(self, blob_service_client, container_name, videos_loaded, folder_limit):
        container_client = blob_service_client.get_container_client(container_name)
        videos_counter = 0
        folder_count = 0

        # Determine folders to load videos from
        if self.single_folder_mode:
            folder_names = [self.specific_folder_name]
        else:
            self.folders = []  # Clear previous entries if any
            folder_names = self.list_blobs_hierarchical(container_client)[:folder_limit]

        for folder_name in folder_names:
            if folder_count >= folder_limit:
                break
            blob_list = container_client.list_blobs(name_starts_with=folder_name)

            for blob in blob_list:
                if videos_loaded is not None and videos_counter >= videos_loaded:
                    break
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                video_data = blob_client.download_blob().readall()
                video = {'path': blob.name, 'data': video_data}
                yield video  # Yield one video at a time
                videos_counter += 1

            self.loaded_folders.append(folder_name)
            folder_count += 1

def test_temporal_four(temporal_four):
    input_frames, frame_order_label, action_label, video_name, frames_canonical_order, selected_frames, input_frames_coordinates, ordered_frames = temporal_four

    visualize_frames(input_frames, selected_frames, frames_canonical_order, video_name, input_frames_coordinates, ordered_frames)

def visualize_frames(input_frames, selected_frames, frames_canonical_order, video_name, input_frames_coordinates, ordered_frames):
    # Reorder the selected_frames to their "original" order based on frames_canonical_order
    inverse_sort_indices = np.argsort(frames_canonical_order)
    shuffled_frames = selected_frames

    # Create a figure and a grid of subplots with an extra row for the column titles
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))

    for i, (cropped_frame, ordered_frame, shuffled_frame) in enumerate(zip(input_frames, ordered_frames, shuffled_frames)):
        cropped_frame = cropped_frame.numpy()

        axs[i, 0].imshow(ordered_frame)
        axs[i, 0].set_title(f'Frame {i}', fontsize=8)
        axs[i, 0].axis('off')

        axs[i, 1].imshow(shuffled_frame)
        axs[i, 1].set_title(f'Frame {frames_canonical_order[i]}', fontsize=8)
        axs[i, 1].axis('off')

        axs[i, 2].imshow(cropped_frame)
        axs[i, 2].set_title(f'Frame {frames_canonical_order[i]}', fontsize=8)
        axs[i, 2].axis('off')

        # New column: overlay cropped frame on uncropped frame
        overlay = shuffled_frame.copy()
        startx, starty = input_frames_coordinates[0][1], input_frames_coordinates[0][0]
        patch_size = cropped_frame.shape[0]
        endx, endy = startx + patch_size, starty + patch_size
        cv2.rectangle(overlay, (startx, starty), (endx, endy), (255, 0, 0), 2)  # Draw red rectangle
        axs[i, 3].imshow(overlay)
        axs[i, 3].set_title(f'Overlay {frames_canonical_order[i]}', fontsize=8)
        axs[i, 3].axis('off')

    # Set a common title for the entire plot
    fig.suptitle(f'Frame Analysis of video {video_name}', fontsize=16)

    # Define the titles for each column
    column_titles = ['Chronological Frames', 'Shuffled Frames', 'Shuffled Cropped Frames', 'Overlay']

    # Set the column titles manually using the text function
    fig.text(0.21, 0.93, column_titles[0], ha='center', va='center', fontsize=10)
    fig.text(0.41, 0.93, column_titles[1], ha='center', va='center', fontsize=10)
    fig.text(0.61, 0.93, column_titles[2], ha='center', va='center', fontsize=10)
    fig.text(0.81, 0.93, column_titles[3], ha='center', va='center', fontsize=10)

    plt.show()

# Example usage:
if __name__ == "__main__":
    sample = BlobSamples(single_folder_mode=False, specific_folder_name=FOLDERNAME)
    print('Processing videos one by one')

    video_generator = sample.load_videos_generator(blob_service_client_instance, CONTAINERNAME, videos_loaded=None, folder_limit=1)
    video_count = 0

    for video in video_generator:
        print(f'Processing video {video_count + 1}')
        video_dataset = PreparedDataset([video])
        temporal_four = PreprocessedTemporalFourData(video_dataset)

        # Save the preprocessed data to Azure Blob Storage
        buffer = io.BytesIO()
        torch.save(temporal_four[0], buffer)  # Save only the first (and only) item
        buffer.seek(0)

        # Upload to Blob Storage
        video_name = video['path'].split('/')[-1].split('.avi')[0]
        blob_client = blob_service_client_instance.get_blob_client(
            container=CONTAINERNAME,
            blob=f"{PREPROCESSEDDATA_FOLDERNAME}/{video_name}_preprocessed.pth"
        )
        blob_client.upload_blob(buffer, overwrite=True)

        print(f"Uploaded preprocessed data for {video_name} to Azure Blob Storage.")
        video_count += 1