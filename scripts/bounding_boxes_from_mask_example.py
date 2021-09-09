import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

from trafo.type_mutating import NDArrayTo3dTensor
from utils import BoundingBoxes


boxes = BoundingBoxes.from_array(np.array([
    [20, 120, 70, 70],
    [50, 20, 30, 50],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]))
mask = boxes.get_mask((200, 120))
plt.imshow(mask)
to_tensor = NDArrayTo3dTensor()
mask_tensor = to_tensor(mask)
transformed_mask = TF.rotate(mask_tensor, 20)[0].numpy()
plt.imshow(transformed_mask)

traf_boxes = BoundingBoxes.from_mask(transformed_mask, 8)
plt.imshow(traf_boxes.get_mask((200, 120)))
