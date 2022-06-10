import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def model_train_plot(history):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    history_dict = history.history
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    loss_ax.plot(train_loss, 'y', label='train loss')
    loss_ax.plot(val_loss, 'r', label='val loss')

    acc_ax.plot(train_acc, 'b', label='train acc')
    acc_ax.plot(val_acc, 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()


def vis(history, name):
    plt.title(f"{name.upper()}", fontsize=20)
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel(f"{name.lower()}", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}", None)
    epochs = range(1, len(value)+1)
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None:
        plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.05, 1.2), fontsize=20, ncol=1)


def plot_history(history):
    key_value = list(set([i.split("val_")[-1]
                     for i in list(history.history.keys())]))
    plt.figure(figsize=(20, 8))
    for idx, key in enumerate(key_value):
        plt.subplot(1, len(key_value), idx+1)
        vis(history, key)
    plt.tight_layout()
    plt.show()


def plot_cf_matrix(cm):
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.figure()
    colormap = sns.color_palette("coolwarm", 12)
    sns.heatmap(cm, cmap=colormap, annot=True, annot_kws={"size": 20})
    plt.xlabel('Prediction', fontsize=20, fontweight='bold',
               horizontalalignment='center')
    plt.ylabel('Target', fontsize=20, fontweight='bold',
               horizontalalignment='center')
    plt.title('Confusion Matrix', fontsize=20, fontweight='bold',
              horizontalalignment='center')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=13)
    plt.show()


def corr_heatmap(corr_df):
    sns.heatmap(corr_df, cmap='viridis')
    plt.show()