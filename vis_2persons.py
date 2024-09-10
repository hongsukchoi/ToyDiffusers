import numpy as np
from PIL import Image
from diffusers.utils import load_image, make_image_grid


sdxl0 = load_image('test_2human_sdxl_posecontrol_promptweight_output_0.png')
sdxl1 = load_image('test_2human_sdxl_posecontrol_promptweight_output_1.png')
sdxl2 = load_image('test_2human_sdxl_posecontrol_promptweight_output_2.png')

det_img = load_image('test_2human_sdxl_control_openpose_input.png')
org_img = load_image('test_2human.jpg')
# Convert copied image to a NumPy array
image_array = np.array(org_img.copy())
# Fill the array with 0 (black for all channels)
image_array.fill(0)
# Convert the array back to a PIL image
black_image = Image.fromarray(image_array)


fine0 = load_image('/home/hongsuk/projects/FineControlNet/outpyt_2poses_many/finecontrolnet_output7.png')
fine1 = load_image('/home/hongsuk/projects/FineControlNet/output_2poses_0/finecontrolnet_output.png')
fine2 = load_image('/home/hongsuk/projects/FineControlNet/outpyt_2poses_many/finecontrolnet_output50.png')
# 63

flux0 = load_image('test_2human_flux_posecontrol_output_0.png')
flux1 = load_image('test_2human_flux_posecontrol_output_1.png')
flux2 = load_image('test_2human_flux_posecontrol_output_3.png')


# image_list = [sdxl0, sdxl1, sdxl2, org_img, det_img, fine]
image_list = [black_image, sdxl0, sdxl1, sdxl2,
              det_img, flux0, flux1, flux2,
              org_img, fine0, fine1, fine2
              ]

grid = make_image_grid(image_list, rows=3, cols=4, resize=256)
grid.save('grid_2people.png')
