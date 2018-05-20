import matplotlib.pyplot as plt
import numpy as np

train_losses = np.load('train_losses.npy')
val_losses = np.load('val_losses.npy')

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()