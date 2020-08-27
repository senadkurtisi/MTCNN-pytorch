from utils.globals import *

from utils.utils import *
from face_detection import *

from PIL import Image

from model import *
from timeit import default_timer as timer


def main():
    # Load the selected image
    img = Image.open(config.img_loc)
    # show_detection(img, [])

    P_net = PNet().to(device)
    R_net = RNet().to(device)
    O_net = ONet().to(device)

    stage_one(P_net, img)

    return img


if __name__ == "__main__":
	a = timer()
	img = main()
	print(timer()-a)


