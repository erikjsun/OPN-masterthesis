import json
import matplotlib.pyplot as plt

# Load mAP data for comparison
def load_map_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['train_mAPs'], data['val_mAPs']
    except FileNotFoundError:
        print(f"File not found: {path}")
        return [], []

# Example comparison script
train_mAPs_pretrained, val_mAPs_pretrained = load_map_data('./saved/map_data_with_pretraining.json')
train_mAPs_no_pretraining, val_mAPs_no_pretraining = load_map_data('./saved/map_data_no_pretraining.json')

# Plot the comparison
plt.figure(figsize=(10, 5))

if train_mAPs_pretrained and val_mAPs_pretrained:
    plt.plot(train_mAPs_pretrained, label='Training mAP (Pretrained)', marker='o')
    plt.plot(val_mAPs_pretrained, label='Validation mAP (Pretrained)', marker='o')

if train_mAPs_no_pretraining and val_mAPs_no_pretraining:
    plt.plot(train_mAPs_no_pretraining, label='Training mAP (No Pretraining)', marker='x')
    plt.plot(val_mAPs_no_pretraining, label='Validation mAP (No Pretraining)', marker='x')

plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Comparison of mAP (Pretrained vs. No Pretraining)')
plt.legend()
plt.grid(True)
plt.savefig('./saved/comparison_plot.png')  # Save the comparison plot
plt.show()