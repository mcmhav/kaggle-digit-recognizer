import matplotlib.pyplot as plt


def create_2d_image(flat_image, image_size):
    img = []
    count = 0
    pixel_row = []
    for pixel in flat_image:
        count += 1

        pixel_row.append(pixel)
        if count % image_size == 0:
            img.append(pixel_row)
            pixel_row = []

    return img


def plot_data(df_train_aligned, df_train_labels):
    image_size = df_train_aligned.shape[1] ** (1 / 2)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        img = create_2d_image(df_train_aligned.loc[i].values, image_size)

        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(df_train_labels.loc[i])

    plt.show()
