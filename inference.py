import os
import cv2
from PIL import Image
import numpy as np
import argparse

from animeinsseg import AnimeInsSeg, AnimeInstances
from animeinsseg.anime_instances import get_color

ckpt = r'models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'
mask_thres = 0.3
instance_thres = 0.3
refine_kwargs = {'refine_method': 'refinenet_isnet'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime Instance Segmentation')
    parser.add_argument('--input', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()

    net = AnimeInsSeg(ckpt, mask_thr=mask_thres, refine_kwargs=refine_kwargs)

    for img_name in os.listdir(args.input):
        img_path = os.path.join(args.input, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        instances: AnimeInstances = net.infer(
            img,
            output_type='numpy',
            pred_score_thr=instance_thres
        )

        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if instances.masks is not None:
            for mask in instances.masks:
                combined_mask = np.logical_or(combined_mask, mask)

        combined_mask = (combined_mask * 255).astype(np.uint8)
        rgba = np.dstack((img_rgb, combined_mask))

        output_image_path = os.path.join(args.output, os.path.splitext(img_name)[0] + '.png')
        Image.fromarray(rgba).save(output_image_path)
        print(f"Transparent background image saved at {output_image_path}")
