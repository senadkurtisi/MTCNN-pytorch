from .globals import *

import torchvision.transforms as transforms
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def prepare_image(img, scale=None):
    ''' Prepares the image for the first
        stage of the inference process.
        Image is resized, normalized and
        transformed into PyTorch tensor.

    Arguments:
        img (PIL image): input image
        scale (float): scale ratio for w&h
    Returns:
        img (torch.Tensor): transformed image
    '''
    if scale:
        import math
        w, h = img.size
        # Calculate shape of rescaled image
        new_w = math.ceil(w * scale)
        new_h = math.ceil(h * scale)
        # Resize the image using bilinear interpolation
        img = img.resize((new_w, new_h), Image.BILINEAR)

    # Prepare the image for the PyTorch nn.Module:
    # 1. & 2. Normalize the image
    # 3. & 4. Transform into PyTorch tensor, PyTorch expects
    # the tensor to be of shape (batch, channels, w, h)
    to_torch = transforms.Compose([transforms.Lambda(lambda x: np.array(x)),
                                   transforms.Lambda(
                                       lambda x: (x - MEAN) / STD),
                                   transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x.unsqueeze(0))
                                   ])
    img = to_torch(img)
    # We want to save as much memory as possible so we don't calculate
    # gradients for the input image
    img.requires_grad = False

    return img.to(device, torch.float)


def create_scales(img):
    ''' Creates scale ratios used to scale the
        input image in order to create the image
        pyramid.

    Arguments:
        img(torch.Tensor): input image
    Returns:
        img_scales (list): scale ratios for the
                           image pyramid
    '''
    img_scales = []
    w, h = img.size
    min_dim = min(h, w)

    # Operations bellow scale the image so that
    # min size we WANT to detect becomes equal
    # to min size we CAN to detect
    ratio = MIN_DETECTION_SIZE / MIN_FACE_SIZE
    min_dim *= ratio

    power = 0
    while min_dim > MIN_DETECTION_SIZE:
        scale = ratio * (MUL_FACTOR**power)
        img_scales.append(scale)

        min_dim *= MUL_FACTOR
        power += 1

    return img_scales


def get_image_patches(bounding_boxes, img, patch_size):
    ''' Extracts patches of the input image based
        on given coordinates of the bboxes.

    Arguments:
        bounding_boxes (numpy.ndarray): detected bboxes
        img (PIL image): input image
        patch_size (int): square size of the image patch
    Returns:
        patches (numpy.ndarray): extracted image patches
    '''
    CHANNELS = 3
    num_boxes = len(bounding_boxes)
    w, h = img.size

    # Handle bbox coordinates which go beyond the limits of the image
    [dy, edy, dx, edx, y, ey, x, ex, h_fixed, w_fixed] = handle_outliers(
        bounding_boxes, w, h, patch_size)

    # Holds every patch
    patches = np.zeros(
        (num_boxes, CHANNELS, patch_size, patch_size), 'float32')

    img_array = np.array(img, 'uint8')
    for ind in range(num_boxes):
        # Create empty array for patch with current h&w
        patch = np.zeros((h_fixed[ind], w_fixed[ind], CHANNELS), 'uint8')

        patch[dy[ind]: (edy[ind] + 1), dx[ind]: (edx[ind] + 1)] = \
            img_array[y[ind]: (ey[ind] + 1), x[ind]: (ex[ind] + 1)]

        # Resize the extracted patch to a desired patch size
        patch = Image.fromarray(patch)
        patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
        patch = np.array(patch, 'float32')

        # Prepare the extracted patch for the PyTorch nn.Module
        patches[ind, :, :, :] = prepare_image(patch).cpu().numpy()

    patches = torch.from_numpy(patches)
    patches.requires_grad = False

    return patches.to(device, torch.float)


def handle_outliers(bboxes, width, height, size):
    ''' Checks if some of the bboxes go beyond
        the limits of the input image. If that
        is the case their coords are limited to
        extreme values. Limits are proposed via
        width and height of the input image.

    Arguments:
        bboxes (numpy.ndarray): detected bboxes
        width (int): width of the input image
        height (int): height of the input image
        size (int): desired patch size
    Returns:
        corrected (list): list of coords necessary
                          for patch extraction. Used
                          in @get_image_patches func.
    '''
    # Acquire coords of bboxes
    x1, y1, x2, y2 = bboxes[:, : 4].T
    # Height and width of each patch
    h, w = (y2 - y1 + 1), (x2 - x1 + 1)

    # Acquire start and end coords of patches
    x, y, ex, ey = x1, y1, x2, y2

    bbox_num = len(bboxes)
    # Starting coords inside the patch
    dx, dy = np.zeros((bbox_num,)), np.zeros((bbox_num,))
    # End coords inside the patch
    edx, edy = w - 1, h - 1

    # If bottom rigth corner goes off limits to the right
    keep_ind = np.where(ex > (width - 1))[0]
    edx[keep_ind] = (w[keep_ind] - 1 - ex[keep_ind]) + (width - 1)
    ex[keep_ind] = width - 1

    # If top left corner goes off limits to the left
    keep_ind = np.where(x < 0)[0]
    dx[keep_ind] = -x[keep_ind]
    x[keep_ind] = 0

    # If bottom right corner goes off limits  downwards
    keep_ind = np.where(ey > (height - 1))[0]
    edy[keep_ind] = (h[keep_ind] - 1 - ey[keep_ind]) + (height - 1)
    ey[keep_ind] = height - 1

    # If top left corner goes off limits upwards
    keep_ind = np.where(y < 0)[0]
    dy[keep_ind] = -y[keep_ind]
    y[keep_ind] = 0

    corrected = [dy, edy, dx, edx, y, ey, x, ex, h, w]
    corrected = [i.astype('int32') for i in corrected]

    return corrected


def show(img, bboxes, landmarks=[]):
    ''' Displays the image and detected
        bboxes and landmarks.

    Arguments:
        img (PIL image): input image
        bboxes (numpy.ndarray): detected bboxes
        landmarks (numpy.ndarray): 5 landmarks for
                                   each detected bbox
                                   5x & 5y coords
    '''
    # Disable the toolbar on the figures
    import matplotlib as mpl
    mpl.rcParams['toolbar'] = 'None'

    # Color of the bboxes
    BOX_COLOR = 'r'
    # Color of landmark points
    LANDMARK_COLOR = 'r'

    img_arr = np.array(img)
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img_arr)

    # Drawing detected bboxes
    for bbox in bboxes:
        # Calculate the width and height of current bbox
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Draw the current bbox
        rect = patches.Rectangle((bbox[0], bbox[1]), w, h,
                                 edgecolor=BOX_COLOR,
                                 facecolor='none')
        ax.add_patch(rect)

    # Drawing detected landmarks
    for landmark in landmarks:
        for ind in range(5):
            # Draw the current landmark
            circle = patches.Ellipse((landmark[ind], landmark[ind + 5]), 3, 3,
                                     edgecolor=LANDMARK_COLOR,
                                     facecolor=LANDMARK_COLOR)
            ax.add_patch(circle)

    # Set the title of the figure
    title = f"Number of detected faces: {len(bboxes)}"
    ax.set_title(title)
    ax.axis('off')
    fig.canvas.set_window_title("MTCNN")
    fig.tight_layout()

    plt.show()
