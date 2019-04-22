import matplotlib.pyplot as plt
import sys


def plot2(log_file):
    loss = [float(f.strip().split()[4]) for f in open(log_file) if "->loss" in f]

    n = len(loss)
    plt.plot(range(n), loss)
    plt.xticks(range(0, n, 8))
    plt.grid()

    # plt.savefig(f'{log_file.split(".")[0]}-plot2.png')
    plt.show()
    plt.close()


def plot(log_file):
    fp = open(log_file)

    _all = fp.read()
    num_epochs = int(_all[_all.find("Epoch"):][:10].split("/")[1].replace(";", ""))
    num_folds = 5
    fp.close()

    train_loss = [float(f.strip().split()[5].split("=")[1]) for f in open(log_file).readlines() if 'loss=' in f]
    val_loss = [float(f.strip().split()[6].split("=")[1]) for f in open(log_file).readlines() if 'loss=' in f]

    assert len(train_loss) == len(val_loss) == num_epochs * num_folds

    n = num_epochs * num_folds

    plt.plot(range(n), train_loss, 'r-')
    plt.plot(range(n), val_loss, 'g-')
    plt.xticks(range(0, n, 2))
    plt.grid()

    plt.savefig(f'{log_file.split(".")[0]}-plot.png')
    plt.close()


plot('m2-v16.log')
