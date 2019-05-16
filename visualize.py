from data_loader import read_data_sets
from networks import capsnet
import matplotlib.pyplot as plt
import numpy as np
import os


def tweak_pose_parameters(output_vectors, d_caps2, min=-0.5, max=0.5, n_steps=11):
    steps = np.linspace(min, max, n_steps)  # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(d_caps2)  # 0, 1, ..., 15
    tweaks = np.zeros([d_caps2, n_steps, 1, 1, 1, d_caps2, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded


# Load data
data_provider = read_data_sets("./data/mnist")
x_test = data_provider.test.images
y_test = data_provider.test.labels

print("Size of:")
print("- Training-set:\t\t{}".format(len(data_provider.train.labels)))
print("- Validation-set:\t\t{}".format(len(data_provider.validation.labels)))
print("- Test-set:\t\t{}".format(len(data_provider.test.labels)))

# Configuration experiment
model_path = "./models/mnist/capsnet/"
my_model = os.path.join(model_path, 'model.cpkt')

# Network definition
net = capsnet.CapsNet(n_class=10, channels=1, is_training=False)

# Reconstruction examples
n_samples = 5
sample_images = data_provider.test.images[:n_samples].reshape([-1, 28, 28, 1])
# Get predictions
preds = net.predict(my_model, sample_images)
predictions = np.argmax(np.squeeze(preds), 1)
# Get reconstructions
reconstructions = net.decode(my_model, sample_images)

sample_images = sample_images.reshape(-1, 28, 28)
reconstructions = reconstructions.reshape([-1, 28, 28])

# Figure: input vs. reconstruction
fig, ax = plt.subplots(2, n_samples)
for row in range(2):
    for index in range(n_samples):
        if row == 0:
            ax[row, index].set_title("Label:" + str(data_provider.test.labels[index]))
            ax[row, index].imshow(sample_images[index], cmap="binary")
        else:
            ax[row, index].set_title("Predicted:" + str(predictions[index]))
            ax[row, index].imshow(reconstructions[index], cmap="binary")
        ax[row, index].axis("off")
plt.show()


# Interpreting the output vectors (of the secondary capsule)
n_steps = 11

caps2_output = net.predict_embedding(my_model, sample_images)

tweaked_vectors = tweak_pose_parameters(caps2_output, d_caps2=net.d_caps2, n_steps=n_steps)
tweaked_vectors_reshaped = tweaked_vectors.reshape([-1, 1, net.n_caps2, net.d_caps2, 1])

tweak_labels = np.tile(data_provider.test.labels[:n_samples], net.d_caps2 * n_steps)

decoder_output = net.tweak_capsule_dimensions(my_model, tweaked_vectors_reshaped, tweak_labels)
tweak_reconstructions = decoder_output.reshape([net.d_caps2, n_steps, n_samples, 28, 28])

for dim in range(3):
    print("Tweaking output dimension #{}".format(dim))
    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    for row in range(n_samples):
        for col in range(n_steps):
            plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
            plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
            if col == int(n_steps/2):
                plt.xticks([], [])
                plt.yticks([], [])
            else:
                plt.axis("off")
    plt.show()
