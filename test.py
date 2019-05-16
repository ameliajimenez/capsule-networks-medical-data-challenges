from data_loader import read_data_sets
from networks import capsnet, lenet, baseline
from sklearn import metrics
import numpy as np
import os

# Load data
data_provider = read_data_sets("./data/mnist/")
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

# Classification performance
n_test = data_provider.test.images.shape[0]
batch_size = 512  # necessary for CapsNet
predictions = np.zeros_like(data_provider.test.labels)
for count, kk in enumerate(range(0, n_test, batch_size)):
    if count == int(n_test / batch_size):
        start = kk
        end = x_test.shape[0]
    else:
        start = kk
        end = kk + batch_size

    n_samples = end - start
    xxtest = x_test[start:end, ...]

    preds = net.predict(my_model, xxtest)
    predictions[start:end] = np.argmax(np.squeeze(preds), 1)

print('Confusion matrix')
print(metrics.confusion_matrix(y_test, predictions))

print('Metrics report')
print(metrics.classification_report(y_test, predictions))

