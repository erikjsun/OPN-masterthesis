from torch.utils.data import Dataset
#saving class to transfer to main.py
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
