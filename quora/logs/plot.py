import matplotlib.pyplot as plt
import sys

log_file = sys.argv[1]

train_loss = [float(f.strip().split()[5].split("=")[1]) for f in open(log_file).readlines() if 'loss=' in f]
val_loss   = [float(f.strip().split()[6].split("=")[1]) for f in open(log_file).readlines() if 'loss=' in f]

num_epochs = 8
num_folds = 5
assert len(train_loss) == len(val_loss) == num_epochs * num_folds

n = num_epochs * num_folds

plt.plot(range(n), train_loss, 'r-')
plt.plot(range(n), val_loss, 'g-')
plt.xticks(range(n))

plt.grid()
plt.show()
