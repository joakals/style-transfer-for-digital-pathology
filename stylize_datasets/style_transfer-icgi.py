## Style transfer following https://arxiv.org/pdf/2102.01678.pdf
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

from tqdm import tqdm
from PIL import Image
from pathlib import Path

import net
from function import adaptive_instance_normalization


def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop != 0:
        transform_list.append(torchvision.transforms.CenterCrop(crop))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def collect_style_files(style_dir: Path, extensions: list):
    # collect style files
    style_dir = style_dir.resolve()
    styles = []
    for ext in extensions:
        styles += list(style_dir.glob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))
    return styles


def read_input_list(input_list: str):
    scan_dirs = []
    with open(input_list, 'r') as f:
        for line in f:
            scan_dir = Path(line.split(',')[0])
            scan_dirs.append(scan_dir)
    return scan_dirs


def stylize_images_in_dir(input_list, style_dir, out_dir, alpha=1, content_size=1024, style_size=256, save_size=512):
    """Style transfer of all scans in input_list
    
    All tiles in one scan are transferred using same style image

    Args:
        input_list: List of paths to scan_dirs containing tiles
    """
    wrk_dir = os.path.dirname(__file__)
    print('working dir: {}'.format(wrk_dir))
    styles = collect_style_files(Path(style_dir), extensions=['png', 'jpeg', 'jpg', 'tif'])
    scan_dirs = read_input_list(input_list)

    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(os.path.join(wrk_dir, 'models/decoder.pth')))
    vgg.load_state_dict(torch.load((os.path.join(wrk_dir, 'models/vgg_normalised.pth'))))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    crop = 0
    content_tf = input_transform(content_size, crop)
    style_tf = input_transform(style_size, 0)

    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    # Sample style images
    sampled_style_images = random.sample(styles, len(scan_dirs))

    # actual style transfer as in AdaIN
    for i, scan_dir in enumerate(tqdm(scan_dirs, desc='style transfer scans')):
        # Assume the folder structure for scan_dir is ../dataset/by_project/train/RES/LABEL/PROJECT/SCANNER/CASEID
        # We want to keep the following from the path by_project/train/RES/LABEL/PROJECT/SCANNER/CASEID
        to_keep = scan_dir.parts[-7:]
        out_dir_scan = out_dir.joinpath(*to_keep)
        # if out_dir_scan.is_dir():
        #     i = 1
        #     out_dir_temp = out_dir_scan
        #     while out_dir_temp.is_dir():
        #         out_dir_temp = out_dir_temp.parent / (out_dir_scan.name + '_{}'.format(i))
        #         i += 1
        #     out_dir_scan = out_dir_temp
        out_dir_scan.mkdir(exist_ok=True, parents=True)

        style_path = sampled_style_images[i]
        style_img = Image.open(style_path).convert('RGB')
        style_img.save(str(out_dir_scan / 'style_image_{}.png'.format(style_path.stem)))
        
        tiles = sorted(list(scan_dir.glob('*.png')))
        for content_path in tqdm(tiles, desc='style transfer {}'.format(scan_dir.name)):
            try:
                content_img = Image.open(content_path).convert('RGB')
                content = content_tf(content_img)
                style = style_tf(style_img)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style, alpha)
                output = output.cpu().squeeze_(0)
                output_img = torchvision.transforms.ToPILImage()(output)
                output_img = output_img.resize((save_size, save_size), Image.LANCZOS)
                out_path = out_dir_scan / content_path.name
                output_img = output_img.save(str(out_path))
                # output = np.array(output_img)

                content_img.close()
            except Exception as err:
                print('skipped stylization of {} because of the following error; {})'.format(content_path, err))
                skipped_imgs.append(str(content_path))
                continue

        style_img.close()    

    if len(skipped_imgs) > 0:
        with open(str(out_dir / 'skipped_imgs.txt'), 'w') as f:
            for item in skipped_imgs:
                f.write("%s\n" % item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_list', required=True, help='List with paths to folders with tiles')
    parser.add_argument('-sd', '--style_dir', required=True, help='Path to directory with images used to transfer style')
    parser.add_argument('-o', '--output_dir', required=True, help='Base output directory')

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    stylize_images_in_dir(args.input_list, args.style_dir, out_dir)


if __name__ == '__main__':
    main()
