from azure.storage.blob import BlobServiceClient, ContainerClient, BlobPrefix
from data.py import BlobSamples, PreparedDataset, PreprocessedTemporalFourData
from data.py import test_video_dataset, test_temporal_four, visualize_frames, visualize_optical_flow

def main():
    STORAGEACCOUNTURL = "https://exjobbssl1863219591.blob.core.windows.net"
    STORAGEACCOUNTKEY = "PuL1QY8bQvIyGi653lr/9CPvyHLnip+cvsu62YAipDjB7onPDxfME156z5/O2NwY0PRLMTZc86/6+ASt5Vts8w=="
    CONTAINERNAME = "exjobbssl"
    FOLDERNAME = "UCF-101/Diving/" #Temporary during the development phase
    #BLOBNAME = "UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"

    blob_service_client_instance = BlobServiceClient(
        account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

    #container_client_instance = blob_service_client_instance.get_container_client(CONTAINERNAME)

    # Usage:
    sample = BlobSamples()

    videos_loaded = 200
    print('loading videos')
    videos = sample.load_videos_into_memory(blob_service_client_instance, CONTAINERNAME, FOLDERNAME, videos_loaded)
    print('videos loaded')
    print("Amount of videos loaded: " + str(len(videos)))

    # Initialize the train/test dataset
    video_dataset = PreparedDataset(videos, trainval='train')

    # Usage:
    amount_of_videos_to_display = 6
    test_video_dataset(video_dataset, amount_of_videos_to_display)

    # Usage:
    amount_of_videos_to_load = 6
    video_dataset = PreparedDataset(videos, trainval='train')
    temporal_four = PreprocessedTemporalFourData(video_dataset, trainval='train')
    test_temporal_four(temporal_four, amount_of_videos_to_load)

if __name__ == '__main__':
    main()