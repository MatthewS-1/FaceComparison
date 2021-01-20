from FaceDataPreprocessing import process, reverse_normalize
from FaceSiameseModel import NUM_EPOCHS
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from FaceSiameseModel import VERSION


colors = ["red", "orange", "yellow", "green", "blue", "purple"]


def analyze_history(hist):
    hist = hist.item()  # hist is originally a np.array; we can get a dictionary from it using .item()
    fig = plt.figure("Model Losses")

    for i, loss in enumerate(hist):
        graph = fig.add_subplot(2, 2, i + 1)
        graph.set_title(loss)
        graph.set_xlabel('epochs')
        graph.set_ylabel('loss value')

        graph.plot([x for x in range(1, NUM_EPOCHS + 1)], hist[loss], color=colors[i])

    plt.show()


def data_from_dir(image_dir):
    assert image_dir[-3:] == "pgm", "must be pgm image"
    image = Image.open(image_dir)

    """if image.size != (128, 128):
        image_data = np.array(image)
        image_data = st.resize(image_data, (128, 128))
        image_data *= 255 / (np.max(image_data) - np.min(image_data))
        image_data = image_data.astype(np.uint8)
        image = Image.fromarray(image_data)
        image.show()"""

    img_data = process(image)
    img_data = img_data.reshape(-1, 128, 128, 1)
    return img_data


def predict(image_dir_tuple, model):
    first_img, second_img = data_from_dir(image_dir_tuple[0]), data_from_dir(image_dir_tuple[1])

    prediction = model.predict([first_img, second_img])
    return prediction


def main():
    model_hist = np.load("face_siamese_" + str(VERSION) + ".npy", allow_pickle=True)
    #analyze_history(model_hist)

    model = load_model("saved_model/face_siamese_" + str(VERSION))

    prediction = predict(("CroppedYale/yaleB01/yaleB01_P00A+000E+00.pgm", "CroppedYale/yaleB01/yaleB01_P00A+000E+00.pgm"),
                         model=model)
    print(prediction)


if __name__ == '__main__':
    main()
