def visualize_network():
    from torchviz import make_dot

    # Assuming `inputs` is a batch of your input data
    inputs = torch.randn(1, 4, 80, 80, 3)  # Adjust the size as needed

    model = CustomOPN()

    # Permute the dimensions to (batch_size, frames, channels, height, width)
    inputs = inputs.permute(0, 1, 4, 2, 3)

    # Flatten the frames and channels dimensions
    inputs = inputs.contiguous().view(-1, inputs.shape[1]*inputs.shape[2], inputs.shape[3], inputs.shape[4])

    # Ensure there is more than one value per channel
    if inputs.shape[0] == 1:
        inputs = inputs.repeat(2, 1, 1, 1)

    outputs = model(inputs)

    # Create the graph
    dot = make_dot(outputs, params=dict(model.named_parameters()))

    # Display the graph
    dot.view()

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_dataset_images(dataset, num_images=5):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        # Assuming each item in dataset gives a batch of images
        img_tensor = dataset[i][0]  # This needs to be the tensor containing the image data

        # Check the number of dimensions and rearrange if necessary
        if img_tensor.ndim == 4 and img_tensor.shape[-1] in {1, 3}:
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # Rearrange from NHWC to NCHW if needed
            img_tensor = img_tensor[0]  # Take the first image of the batch

        # Handle single channel images (grayscale)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)  # Remove channel dimension if it's grayscale

        img = TF.to_pil_image(img_tensor)
        axs[i].imshow(img, cmap='gray' if img_tensor.shape[0] == 1 else None)
        axs[i].axis('off')
    plt.show()