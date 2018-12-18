import face_recognition
import matplotlib.patches as ptc
import matplotlib.pyplot as plt

image = face_recognition.load_image_file("testing/2.jpg")
faceLocations = face_recognition.face_locations(image)

for (top, right, bottom, left) in  faceLocations:

    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")

    rectangle1 = ptc.Rectangle((left, top),
                               (right - left),
                               (bottom - top),
                               linewidth=1,
                               edgecolor='r',
                               facecolor='none')
    ax.add_patch(rectangle1)
    plt.show()