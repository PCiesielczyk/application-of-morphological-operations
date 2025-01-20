import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import img_as_float, io, color
from skimage.segmentation import (
    morphological_chan_vese,
    checkerboard_level_set, inverse_gaussian_gradient, morphological_geodesic_active_contour,
)

INTERVAL_ACWE = 200
INTERVAL_GAC = 25

SQUARE_SIZE_ACWE = 6

NUM_ITER_ACWE = 35
NUM_ITER_GAC = 250

SMOOTHING = 1

INPUT_NAME = "seastar.png"
OUTPUT_NAME = "s_3"


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# ============== Morphological ACWE ===============
# image = img_as_float(data.camera())
image = img_as_float(io.imread(INPUT_NAME))

if image.shape[-1] == 4:
    image = image[..., :3]

if len(image.shape) == 3:
    image_gray = color.rgb2gray(image)
else:
    image_gray = image.copy()

init_ls = checkerboard_level_set(image_gray.shape, SQUARE_SIZE_ACWE)
evolution_acwe = []
callback = store_evolution_in(evolution_acwe)
ls_acwe = morphological_chan_vese(
    image_gray, num_iter=NUM_ITER_ACWE, init_level_set=init_ls, smoothing=SMOOTHING, iter_callback=callback
)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap="gray")
ax.set_axis_off()
ax.set_title("Morphological ACWE segmentation", fontsize=12)

contour_acwe = [ax.contour(evolution_acwe[0], [0.5], colors='r')]


def update_acwe(frame):
    for c in contour_acwe[0].collections:
        c.remove()
    contour_acwe[0] = ax.contour(evolution_acwe[frame], [0.5], colors='r')
    return contour_acwe[0].collections


ani = FuncAnimation(
    fig, update_acwe, frames=len(evolution_acwe), interval=INTERVAL_ACWE, blit=False
)

ani.save(OUTPUT_NAME + "_acwe.gif")
print("ani_acwe.gif saved!")

# ========= Morphological GAC ==============

gimage = inverse_gaussian_gradient(image_gray)

init_ls = np.zeros(image_gray.shape, dtype=np.int8)
init_ls[1:-1, 1:-1] = 1

evolution_gac = []
callback = store_evolution_in(evolution_gac)
ls_gac = morphological_geodesic_active_contour(
    gimage,
    num_iter=NUM_ITER_GAC,
    init_level_set=init_ls,
    smoothing=1,
    balloon=-1,
    threshold=0.69,
    iter_callback=callback,
)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap="gray")
ax.set_axis_off()
ax.set_title("Morphological GAC segmentation", fontsize=12)

contour_gac = [ax.contour(evolution_gac[0], [0.5], colors='r')]


def update_gac(frame):
    for c in contour_gac[0].collections:
        c.remove()
    contour_gac[0] = ax.contour(evolution_gac[frame], [0.5], colors='r')
    return contour_gac[0].collections


ani = FuncAnimation(
    fig, update_gac, frames=len(evolution_gac), interval=INTERVAL_GAC, blit=False
)

ani.save(OUTPUT_NAME + "_gac.gif")
