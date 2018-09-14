import matplotlib.pyplot as plt

def plot_graphs(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(history.epoch, history.history["loss"], label="Train loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(132)
    plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
    plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
    plt.legend()
    plt.subplot(133)
    plt.plot(history.epoch, history.history["mean_iou"], label="Train iou")
    plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
    plt.legend()
    plt.show()