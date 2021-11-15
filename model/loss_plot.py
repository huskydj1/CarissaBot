import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
train_losses = [0.04907, 0.03614, 0.03240, 0.02983, 0.02794, 0.02663, 0.02547, 0.02453, 0.02341, 0.02250, 0.02161, 0.02069, 0.01972]
test_losses = [0.03968, 0.03357, 0.03394, 0.02950, 0.02880, 0.03210, 0.02717, 0.02929, 0.02719, 0.02544, 0.02559, 0.02621, 0.02510]

plt.plot(x, train_losses, label='Train Loss')
plt.plot(x, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig('loss_plot.png')