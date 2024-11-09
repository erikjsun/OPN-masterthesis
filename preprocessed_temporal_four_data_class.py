#saving class to transfer to main.py
from torch.utils.data import Dataset
import imageio.v3 as iio
import cv2
import numpy as np
import torch
import random
from PIL import Image

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
