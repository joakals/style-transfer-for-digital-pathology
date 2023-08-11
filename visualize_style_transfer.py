import argparse
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


def plot_transferred_next_to_style_image(transferred_img_paths: list, style_img_paths: list, output_path: str):
    """Plots the transferred images next to the style image used for the style transfer

    Args:
        transferred_img_paths: paths to images that have been style transferred7
        style_img_paths: paths to style images used for style transfer
    """
    assert len(transferred_img_paths) == len(style_img_paths)

    f, axarr = plt.subplots(2, len(transferred_img_paths))
    for i, (tr_img, st_img) in enumerate(zip(transferred_img_paths, style_img_paths)):
        transfer_img = Image.open(str(tr_img)).convert('RGB')
        style_img = Image.open(str(st_img)).convert('RGB')

        axarr[0, i].imshow(transfer_img)
        axarr[1, i].imshow(style_img)

    f.savefig(output_path)


def collect_images(base_dir, tile_name):
    style_transferred_tiles = list(base_dir.rglob('*' + tile_name + '*'))
    style_images = [(x.parents[0] / 'style_image.png') for x in style_transferred_tiles]
    assert all([s.is_file() for s in style_images])
    return style_transferred_tiles, style_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir')
    parser.add_argument('--out_dir')
    parser.add_argument('--tile_names', nargs='+')

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    for tile_name in args.tile_names:
        output_path = out_dir / (tile_name + '.png')
        transferred_images, style_images = collect_images(base_dir, tile_name)
        plot_transferred_next_to_style_image(transferred_images, style_images, output_path)


if __name__ == '__main__':
    main()
