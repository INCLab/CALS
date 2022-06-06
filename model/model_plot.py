import matplotlib.pyplot as plt
import seaborn as sns


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


def corr_heatmap(corr_df):
    sns.heatmap(corr_df, cmap='viridis')
    plt.show()