import numpy as np
from PIL import Image
from diffusers.utils import load_image, make_image_grid

det_img = load_image('test_3human_flux_control_openpose_input.png')
org_img = load_image('test_3human.jpg')
# Convert copied image to a NumPy array
image_array = np.array(org_img.copy())
# Fill the array with 0 (black for all channels)
image_array.fill(0)
# Convert the array back to a PIL image
black_image = Image.fromarray(image_array)


fine0 = load_image('/home/hongsuk/projects/FineControlNet/finecontrolnet_output52.png')
fine1 = load_image('/home/hongsuk/projects/FineControlNet/finecontrolnet_output18.png')
fine2 = load_image('/home/hongsuk/projects/FineControlNet/finecontrolnet_output82.png')
# 52
# 18

flux0 = load_image('test_3human_flux_posecontrol_output_1.png')
flux1 = load_image('test_3human_flux_posecontrol_output_2.png')
flux2 = load_image('test_3human_flux_posecontrol_output_11.png')

# image_list = [sdxl0, sdxl1, sdxl2, org_img, det_img, fine]
image_list = [
              det_img, flux0, flux1, flux2,
              org_img, fine0, fine1, fine2
              ]

grid = make_image_grid(image_list, rows=2, cols=4, resize=256)
grid.save('grid_3people.png')
