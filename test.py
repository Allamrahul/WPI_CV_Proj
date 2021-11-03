import os
import torch
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import Transforms as T
cmap = plt.cm.viridis

IMAGE_HEIGHT, IMAGE_WIDTH = 1024, 1024  # raw image size
output_size = (224, 224)  # transform output size

home_path = os.path.abspath(os.path.join(os.getcwd()))

test_path = os.path.join(home_path, "data", "test")

print(test_path)

images = glob.glob(os.path.join(test_path, "*.jpg"))

print(images)

checkpoint = torch.load(os.path.join(home_path, "results", "mobilenet-nnconv5dw-skipadd-pruned.pth.tar"))

model = checkpoint['model']


def display_t(ip):
    # pil = transforms.ToPILImage()(ip)
    # pil_img = Image.open(pil)
    # pil_img.show()
    plt.imshow(transforms.ToPILImage()(ip), interpolation="bicubic")
    plt.show()


def validationTransform(rgb):
    first_resize = tuple(map(int, list((250.0 / IMAGE_HEIGHT) * np.array([IMAGE_HEIGHT, IMAGE_WIDTH]))))
    transform = T.Compose([
        # T.Resize(first_resize),
        # T.CenterCrop((228, 304)),
        T.Resize(output_size),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255

    return rgb_np

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)

def main():

    for i,image in enumerate(images):
        img = Image.open(image)
        np_arr_img = np.asarray(img)  # converts PIL image into numpy array; shape 1024 x 1024 x 3
        test_transform_op = validationTransform(np_arr_img)

        to_tensor = T.ToTensor()
        input_tensor = to_tensor(test_transform_op)
        print(input_tensor.dim())
        while input_tensor.dim() <= 3:
            input_tensor = input_tensor.unsqueeze(0)

        input = input_tensor.cuda()
        print(input.shape)
        with torch.no_grad():
            pred = model(input)

        rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
        depth_input_cpu = np.squeeze(input.cpu().numpy())
        depth_pred_cpu = np.squeeze(pred.data.cpu().numpy())

        # d_min = min(np.min(depth_input_cpu), np.min(depth_pred_cpu))
        # d_max = max(np.max(depth_input_cpu), np.max(depth_pred_cpu))
        depth_input_col = colored_depthmap(depth_input_cpu)
        depth_pred_col = colored_depthmap(depth_pred_cpu)

        img_merge = np.hstack([rgb, depth_pred_col])
        image_name = "Depth_map_" + str(i) + ".jpg"
        save_image(img_merge, image_name)


        #convert_tensor = transforms.ToTensor()
        #pil_to_tensor = convert_tensor(img).unsqueeze_(0)
        # img.show()
        # print(pil_to_tensor.shape)


if __name__ == "__main__":
    main()

