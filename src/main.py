from utils.globals import *

from utils.image_utils import show
from face_detection import *

from PIL import Image

from model import *
from timeit import default_timer as timer


def main():
    start = timer()
    # Load the selected image
    img = Image.open(config.img_loc)

    # Create necessary CNNs
    P_net = PNet().to(device)
    R_net = RNet().to(device)
    O_net = ONet().eval().to(device)

    # Peforms the first stage of the detection process
    bboxes = stage_one(P_net, img)
    # Peforms the second stage of the detection process
    bboxes = stage_two(R_net, bboxes, img)
    # Peforms the third stage of the detection process
    bboxes, landmarks = stage_three(O_net, bboxes, img)

    total_time = timer() - start
    print(f"Detection time:{round(total_time, 2)} s")
    print(f"Number of detected faces: {len(bboxes)}")

    show(img, bboxes, landmarks)


if __name__ == "__main__":
    main()
