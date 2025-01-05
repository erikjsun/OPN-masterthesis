#DATA_PREP_PY
# 1. IMPORTS
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
import gc
import traceback

# 2. CONSTANTS
STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
CONTAINERNAME = "exjobbssl"
FOLDERNAME = "UCF-101/HighJump/"
PREPROCESSEDDATA_FOLDERNAME = "ucf-preprocessed-data-500"
BATCH_SIZE = 500
DEFAULT_FOLDER_LIMIT = 101

# 3. CLASSES
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
                if depth > 0:  # Avoid adding the root directory if not needed
                    self.folders.append(blob.name)
                self.list_blobs_hierarchical(container_client, blob.name, depth+1)
        return self.folders

    def load_videos_generator(self, blob_service_client, container_name, videos_loaded=None, folder_limit=DEFAULT_FOLDER_LIMIT):
        print("\nInitializing video generator...")
        container_client = blob_service_client.get_container_client(container_name)
        videos_counter = 0
        folder_count = 0

        # Determine folders to load videos from
        if self.single_folder_mode:
            folder_names = [self.specific_folder_name]
            print(f"\nSingle folder mode: loading from {self.specific_folder_name}")
        else:
            self.folders = []  # Clear previous entries if any
            all_folders = self.list_blobs_hierarchical(container_client)
            folder_names = all_folders[:folder_limit]  # Use the actual folder_limit parameter
            total_folders = len(folder_names)  # Store total folders to process
            print(f"\nFound {len(all_folders)} total folders in UCF-101")
            print(f"Will process {total_folders} folders:")
            for i, folder in enumerate(folder_names, 1):
                print(f"{i}. {folder.replace('UCF-101/', '')}")
        
        for i, folder_name in enumerate(folder_names, 1):
            folder_count += 1
            if folder_count > folder_limit:
                break
            print(f"\nProcessing folder {i}/{total_folders}: {folder_name}")  # Use i and total_folders
            blob_list = list(container_client.list_blobs(name_starts_with=folder_name))
            print(f"Found {len(blob_list)} videos in folder")

            for blob in blob_list:
                if videos_loaded is not None and videos_counter >= videos_loaded:
                    break
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                video_data = blob_client.download_blob().readall()
                video = {'path': blob.name, 'data': video_data}
                yield video
                videos_counter += 1
                print(f"\rProcessed video {videos_counter} in current folder", end="")
            print("\n")  # Add extra newline after processing each folder

            self.loaded_folders.append(folder_name)
    
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

class PreparedDataset(Dataset):
    def __init__(self, videos, batch_size=10, trainval='train'):
        self.video_names_train = []
        self.video_names_test = []
        self.action_labels_train = []
        self.action_labels_test = []
        self.predata_train = []
        self.predata_test = []
        self.videos = videos
        self.trainval = trainval
        self.batch_size = batch_size

        # Read the classes file
        with open('classInd.txt', 'r') as f:
            classes = f.readlines()
        classes = [c.strip().split(' ', 1)[1] for c in classes]
        self.class_to_id = {c: i for i, c in enumerate(classes)}  # Create a dictionary mapping class names to ids

        # Load train and test paths from files
        train_paths = {}
        test_paths = {}
        with open('trainlist1.txt', 'r') as f:
            for line in f:
                path, label = line.strip().split(' ')
                train_paths[path] = int(label) - 1  # Zero-indexed labels

        with open('testlist1.txt', 'r') as f:
            test_paths = {line.strip() for line in f}

        # Create video batches to load in smaller chunks
        self.video_batches_train, self.video_batches_test = self.create_video_batches(train_paths, test_paths)

    def create_video_batches(self, train_paths, test_paths):
        """
        Split videos into smaller batches based on the batch_size.
        Each batch contains a tuple of (video_name, video, label) for both train and test sets.
        """
        video_batches_train = []
        video_batches_test = []
        train_batch = []
        test_batch = []

        for video in self.videos:
            path = video['path'][len('UCF-101/'):]  # Remove the 'UCF-101/' prefix from the video path
            video_name = path.split('/')[1].split('.avi')[0].replace('v_', '')  # Extract the video name
            
            if path in train_paths:
                label = train_paths[path]  # Get the label from the dictionary
                train_batch.append((video_name, video, label))  # Append the train video and label
                self.video_names_train.append(video_name)
                self.action_labels_train.append(label)
                self.predata_train.append(video)
            elif path in test_paths:
                class_name = path.split('/')[0]
                label = self.class_to_id[class_name]
                test_batch.append((video_name, video, label))  # Append test video with label
                self.video_names_test.append(video_name)
                self.action_labels_test.append(label)
                self.predata_test.append(video)

            # When batch size is reached for train data, store the batch and start a new one
            if len(train_batch) == self.batch_size:
                video_batches_train.append(train_batch)
                train_batch = []

            # When batch size is reached for test data, store the batch and start a new one
            if len(test_batch) == self.batch_size:
                video_batches_test.append(test_batch)
                test_batch = []

        # Append the last train batch if it's not empty
        if train_batch:
            video_batches_train.append(train_batch)

        # Append the last test batch if it's not empty
        if test_batch:
            video_batches_test.append(test_batch)

        return video_batches_train, video_batches_test

    def __getitem__(self, index):
        """
        Returns a batch of videos for either the train or test set. 
        Each batch contains:
        - Video name
        - Video data
        - Action label
        """
        if self.trainval == 'train':
            return self.video_batches_train[index]
        else:
            return self.video_batches_test[index]

    def __len__(self):
        """
        Return the number of batches for either the train or test set.
        """
        if self.trainval == "train":
            return len(self.video_batches_train)
        else:
            return len(self.video_batches_test)

class PreprocessedTemporalFourData(Dataset):
    def __init__(self, dataset, trainval='train', pixel_mean = 96.5, patch_size=160):
        self.dataset = dataset
        self.trainval = trainval
        self.pixel_mean = pixel_mean
        self.patch_size = patch_size
        
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
        weights, flows = self.compute_optical_flow_weights(frames)
        
        # Select the four frames from the video based on the weights of the optical flow – random order
        indices = np.random.choice(frames.shape[0], size=4, replace=False, p=weights)
        selected_frames = frames[indices]

        # Normalize the indices to their respective order
        ranked_indices = indices.argsort().argsort()
        
        #saving the ordered frames for visualization
        ordered_indices = np.sort(indices)
        ordered_frames = frames[ordered_indices]

        # Create label list for the frame order
        frame_order_label, frames_canonical_order = self.get_frame_order_label(ranked_indices)

        # Applying random mirroring to all selected frames at once
        mirror = random.randint(0,1)
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
        #print("Shape of preprocessed_frames:", preprocessed_frames.shape)
        return preprocessed_frames, frame_order_label, action_label, video_name, frames_canonical_order, selected_frames, preprocessed_frames_coordinates, ordered_frames

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
        # Downsample the frames to reduce the optical flow computation (complexity is O(N^2) where N is the number of pixels)
        downsampled_frames = [cv2.resize(frame, (160, 80)) for frame in frames]

        # Compute the optical flow between frames
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in downsampled_frames] #TODO Revert to downsampled_frames, figuring out the optimal downsampling size?
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

    def select_best_patch(self, selected_frames):
        # Compute the optical flow between selected_frames
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in selected_frames]
        flows = [cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(gray_frames)-1)]

        # Compute the magnitude of the optical flow
        magnitudes = [np.sqrt(flow[...,0]**2 + flow[...,1]**2) for flow in flows]

        # Compute the sum of optical flow magnitudes for the selected frames, since the best patch will be found between them
        summed_flow_magnitude = sum(magnitudes)

        # Initialize the best patch and its motion sum
        best_patch = None
        best_motion_sum = -1

        # Define the margin from the frame's edge where patches should not be selected
        margin = 30

        # Slide the patch over the summed frame, and find the patch with the largest motion sum. 
        # The patch cannot be placed outside the frame, and should be at least 'margin' pixels away from the frame's edge.
        for i in range(margin, summed_flow_magnitude.shape[0] - self.patch_size - margin + 1):
            for j in range(margin, summed_flow_magnitude.shape[1] - self.patch_size - margin + 1):
                # Compute the motion sum for the current patch
                current_motion_sum = summed_flow_magnitude[i:i+self.patch_size, j:j+self.patch_size].sum()

                # If the current motion sum is larger than the best one, update the best patch
                if current_motion_sum > best_motion_sum:
                    best_patch = (i, j)
                    best_motion_sum = current_motion_sum
     
        return best_patch

    def preprocess_frames(self, selected_frames, best_patch):
        # Define the spatial jittering distance (NOTE: cannot be larger than margin)
        sjdis = 20
        startx, starty = best_patch

        preprocessed_frames = []
        preprocessed_frames_coordinates = []
        for i, frame in enumerate(selected_frames):            
            #TODO # Subtract the mean: #frame = frame - self.mean
            # Apply channel splitting
            #frame = self.channel_splitting(frame)

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
        sj_frame = frame[newx:newx+self.patch_size, newy:newy+self.patch_size]

        # Return the spatially jittered frame and its starting coordinates
        return sj_frame, (newx, newy)

    def channel_splitting(self, frame):
        # Choose a random color channel for the channel splitting
        rgb = random.randint(0,2)
        frame = frame[:,:,rgb]

        # Duplicate the chosen channel to the other two channels
        frame = np.stack((frame,)*3, axis=2)

        # Subtract the pixel_mean
        frame = frame.astype(float) - self.pixel_mean

        # Normalize the frame to the range [0, 1]
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))

        # Scale up to the range [0, 255]
        frame = frame * 255

        return frame

# 4. HELPER FUNCTIONS
def get_memory_usage():
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Memory in MB

def log_memory(message):
    print(f"{message} - Memory usage: {get_memory_usage():.2f} MB")

def test_temporal_four(temporal_four, n):
    input_frames, frame_order_label, action_label, video_name, frames_canonical_order, selected_frames, input_frames_coordinates, ordered_frames = temporal_four[2]
    print('testing')
    visualize_frames(input_frames, selected_frames, frames_canonical_order, video_name, input_frames_coordinates, ordered_frames)
    #visualize_optical_flow(flows, indices) # TODO add method to get_flows&indices

def visualize_frames(input_frames, selected_frames, frames_canonical_order, video_name, input_frames_coordinates, ordered_frames):
    # Reorder the select_frames to their "original" order based on frames_canonical_order
    #ordered_uncropped_frames = shuffled_uncropped_frames[np.sort(frames_canonical_order)]
    inverse_sort_indices = np.argsort(frames_canonical_order)
    shuffled_frames = selected_frames#[inverse_sort_indices]

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

def process_batch_fully_extracted(batch, batch_count, blob_service_client_instance):
    """
    Processes a batch of raw videos, creates a PreprocessedTemporalFourData-like logic,
    and then saves the final 4-frame items (8-tuple) into a flat list.
    """
    try:
        print(f"\n[FullyExtracted] Starting batch {batch_count} processing...")
        print(f"[FullyExtracted] Batch size: {len(batch)} videos")

        # 1) Create the "video dataset" just like before
        video_dataset = PreparedDataset(batch, trainval='train')  
        #   This splits videos into train/test, but if you only want to store “trainval=’train’” data
        #   that’s fine. Or do it for both if you prefer.

        # 2) Create "temporal dataset" that implements all the random logic
        temporal_four_dataset = PreprocessedTemporalFourData(
            dataset=video_dataset, trainval='train'
        )

        # 3) Now we will call __getitem__ on each index to get the final 8-tuple
        print("[FullyExtracted] Extracting final 4-frame samples from the entire dataset...")
        final_samples = []
        dataset_length = len(temporal_four_dataset)
        print(f"[FullyExtracted] dataset_length = {dataset_length}")

        for idx in range(dataset_length):
            try:
                sample_8 = temporal_four_dataset[idx]
                # sample_8 is typically:
                # (
                #   preprocessed_frames,         # Tensor
                #   frame_order_label,           # Tensor
                #   action_label,                # Tensor
                #   video_name,                  # str
                #   frames_canonical_order,      # Tensor
                #   selected_frames,             # ndarray
                #   preprocessed_frames_coords,  # ...
                #   ordered_frames               # ...
                # )
                final_samples.append(sample_8)

            except Exception as e:
                print(f"[FullyExtracted] Error extracting index={idx}: {str(e)}")
                traceback.print_exc()
                continue

        print(f"[FullyExtracted] Collected {len(final_samples)} final samples.")

        # 4) Save final_samples to memory buffer
        buffer = io.BytesIO()
        torch.save(final_samples, buffer, pickle_protocol=5)
        buffer.seek(0)

        # 5) Upload buffer to Azure
        blob_client = blob_service_client_instance.get_blob_client(
            container=CONTAINERNAME,
            blob=f"{PREPROCESSEDDATA_FOLDERNAME}/ucf101_preprocessed_fullyextracted_batch_{batch_count}.pth"
        )
        blob_client.upload_blob(buffer, overwrite=True)

        # Clean up
        del video_dataset
        del temporal_four_dataset
        del final_samples
        del buffer
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[FullyExtracted] Completed batch {batch_count}")
        return True

    except Exception as e:
        print(f"[FullyExtracted] Error in batch {batch_count}: {str(e)}")
        traceback.print_exc()
        return False

# 5. MAIN EXECUTION
if __name__ == "__main__":
    try:
        print("Starting data preprocessing pipeline...")
        log_memory("Initial memory usage")
        
        # Initialize blob service
        print("Initializing Azure Blob Storage connection...")
        blob_service_client_instance = BlobServiceClient(
            account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
        container_client_instance = blob_service_client_instance.get_container_client(CONTAINERNAME)
        
        # Option 2: For multiple folders (default)
        sample = BlobSamples(single_folder_mode=False, specific_folder_name="")
        
        video_generator = sample.load_videos_generator(
            blob_service_client_instance,
            CONTAINERNAME,
            videos_loaded=None,
            folder_limit=DEFAULT_FOLDER_LIMIT
        )

        # Process videos in batches
        print("\nStarting batch processing...")
        batch = []
        batch_count = 0
        video_count = 0
        failed_videos = []
        
        for video in video_generator:
            try:
                batch.append(video)
                video_count += 1
                
                if len(batch) == BATCH_SIZE:
                    batch_count += 1
                    success = process_batch_fully_extracted(batch, batch_count, blob_service_client_instance)
                    if not success:
                        print(f"Failed to process batch {batch_count}")
                        # Store failed videos for retry
                        failed_videos.extend(batch)
                    batch = []
                    log_memory(f"After processing batch {batch_count}")
                    
            except Exception as e:
                print(f"\nError processing video {video_count}: {str(e)}")
                failed_videos.append(video)
                continue
        
        # Process remaining videos
        if batch:
            try:
                batch_count += 1
                success = process_batch_fully_extracted(batch, batch_count, blob_service_client_instance)
                if not success:
                    failed_videos.extend(batch)
            except Exception as e:
                print(f"\nError processing final batch: {str(e)}")
                failed_videos.extend(batch)
        
        # Report results
        print("\nPreprocessing complete!")
        print(f"Total videos processed: {video_count}")
        print(f"Total batches created: {batch_count}")
        print(f"Failed videos: {len(failed_videos)}")
        
        log_memory("Final memory usage")
        
    except Exception as e:
        print(f"\nCritical error in main process: {str(e)}")
    finally:
        print("\nProcess finished.")

    #amount_of_videos_to_load = 6
    #test_temporal_four(temporal_four, amount_of_videos_to_load)