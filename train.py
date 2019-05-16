from data_loader import read_data_sets
from networks import capsnet, lenet, baseline


# Load data
# Experiment 1: Limited amount of data. For example percentage_train=5 to use 5% of balanced training data.
# Experiment 2: Class-imbalance. For example unbalance=True to reduce to 20% the digits 0 and 8 (by default),
# to specify other configurations change the values in unbalance_dict={"percentage": 20, "label1": 0, "label2": 8}.
# Experiment 3: Data augmentation.

data_provider = read_data_sets("data/mnist")

print("Size of:")
print("- Training-set:\t\t{}".format(len(data_provider.train.labels)))
print("- Validation-set:\t\t{}".format(len(data_provider.validation.labels)))
print("- Test-set:\t\t{}".format(len(data_provider.test.labels)))

# Configuration experiment
model_path = "./models/mnist/capsnet/"

# optimizer parameters
name_opt = "adam"
learning_rate = 1e-3
opt_kwargs = dict(learning_rate=learning_rate)
# training parameters
batch_size = 128
n_epochs = 5

# Network definition
net = capsnet.CapsNet(n_class=10, channels=1, is_training=True)

# Training
trainer = capsnet.Trainer(net, optimizer=name_opt, batch_size=batch_size, opt_kwargs=opt_kwargs)

path = trainer.train(data_provider, model_path, n_epochs=n_epochs)
