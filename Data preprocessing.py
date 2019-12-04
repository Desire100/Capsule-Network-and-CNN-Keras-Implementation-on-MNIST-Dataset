# download training and test data from mnist and reshape it


(x_train, y_orig_train), (x_test, y_orig_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(-1,28,28,1)
y_train = np.array(to_categorical(y_orig_train.astype('float32')))

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(-1,28,28,1)
y_test = np.array(to_categorical(y_orig_test.astype('float32')))

x_output = x_train.reshape(-1,784)
X_valid_output = x_test.reshape(-1,784)

n_samples = 5

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = x_test[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary")
    plt.title("Label:" + str(y_orig_test[index]))
    plt.axis("off")

plt.show()