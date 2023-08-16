# Check the style images to exclude the images that are very monochrome with little contrast
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageStat


def get_stat_for_one_style_image(style_image_path):
    style_img = Image.open(str(style_image_path)).convert('RGB')
    style_img = style_img.convert('HSV')
    stats = ImageStat.Stat(style_img)

    results = {}
    for band, name in enumerate(style_img.getbands()): 
        results['{}_extrema'.format(name)] = stats.extrema[band]
        results['{}_std'.format(name)] = stats.stddev[band]
        # print(f'Band: {name}, min/max: {stats.extrema[band]}, stddev: {stats.stddev[band]}')

    return results


def get_hsv_stat_for_one_style_image(style_image_path):
    style_img = Image.open(str(style_image_path)).convert('RGB')
    style_img = style_img.convert('HSV')
    h, s, v = style_img.split()

    hue_range = np.percentile(h, 95) - np.percentile(h, 5)
    saturation_range = np.percentile(s, 95) - np.percentile(s, 5)
    value_range = np.percentile(v, 95) - np.percentile(v, 5)

    hue_hist, bin_edges = np.histogram(h, bins=np.linspace(0, 256, 11))
    sat_mean = np.mean(s)
    val_mean = np.mean(v)

    results = {'hue_hist': hue_hist, 'hue_range': hue_range, 'sat_range': saturation_range, 'val_range': value_range,
               'sat_mean': sat_mean, 'val_mean': val_mean}
    return results


def filter_images(stats_df, out_dir, filter_thresholds=(25, 100)):
    filtered_df = stats_df.loc[(stats_df['mean_std'] < filter_thresholds[0]) | (stats_df['mean_std'] > filter_thresholds[1])]
    to_filter_out = filtered_df['style_image'].values.tolist()
    for img in to_filter_out:
        img = Path(img)
        out_file = out_dir / img.name
        print('Moving image {} from {} to {}'.format(img.name, str(img.parent), str(out_dir)))
        shutil.move(str(img), str(out_file))


def filter_images_by_hsv(stats_df, out_dir, hue_range_filter, sat_range_filter, val_range_filter, sat_mean_filter, val_mean_filter):
    to_filter_out = []

    for metric, filter_name in zip(['hue_range', 'sat_range', 'val_range', 'sat_mean', 'val_mean'],
                                   [hue_range_filter, sat_range_filter, val_range_filter, sat_mean_filter, val_mean_filter]):
        filtered_df = stats_df.loc[stats_df[metric] < filter_name]
        to_filter_out += filtered_df['style_image'].values.tolist()

    to_filter_out_set = set(to_filter_out)
    print('found {} images to filter'.format(len(to_filter_out_set)))

    for img in to_filter_out_set:
        img = Path(img)
        out_file = out_dir / img.name
        print('Moving image {} from {} to {}'.format(img.name, str(img.parent), str(out_dir)))
        shutil.move(str(img), str(out_file))


def main():
    wrk_dir = os.path.dirname(__file__)

    style_image_dir = Path(os.path.join(wrk_dir, 'style_images/train_2'))
    # style_image_dir = Path(os.path.join(wrk_dir, 'images/test_crc_tiles'))
    all_style_images = list(style_image_dir.glob('*.jpg'))
    # all_style_images = list(style_image_dir.glob('*.png'))

    collect_results = {'style_image': [], 'hue_hist': [], 'hue_range': [], 'sat_range': [], 'val_range': [],
                       'sat_mean': [], 'val_mean': []}
    for style_image in tqdm(all_style_images):
        results_img = get_hsv_stat_for_one_style_image(style_image)

        collect_results['style_image'].append(str(style_image))
        for k in ['hue_hist', 'hue_range', 'sat_range', 'val_range', 'sat_mean', 'val_mean']:
            collect_results[k].append(results_img[k])

    df = pd.DataFrame.from_dict(collect_results)
    df['tmp_file_name'] = df['style_image'].apply(lambda x: Path(x).stem).apply(pd.to_numeric)
    df.sort_values(by='tmp_file_name', inplace=True)
    df.drop('tmp_file_name', axis=1, inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_hsv.csv')
    df.to_csv(output_path, index=False)

    # order by lowest hue_range
    df.sort_values(by='hue_range', inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_ordered_by_hue_range.csv')
    df.to_csv(output_path, index=False)

    # order by lowest saturation_range
    df.sort_values(by='sat_range', inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_ordered_by_sat_range.csv')
    df.to_csv(output_path, index=False)

    # order by lowest value_range
    df.sort_values(by='val_range', inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_ordered_by_val_range.csv')
    df.to_csv(output_path, index=False)

    # order by lowest saturation_mean
    df.sort_values(by='sat_mean', inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_ordered_by_sat_mean.csv')
    df.to_csv(output_path, index=False)

    # order by lowest value_mean
    df.sort_values(by='val_mean', inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_ordered_by_val_mean.csv')
    df.to_csv(output_path, index=False)


    # filter images
    out_dir = style_image_dir / 'filtered_images'
    out_dir.mkdir(exist_ok=True)
    filter_images_by_hsv(df, out_dir, hue_range_filter=20, sat_range_filter=50, val_range_filter=75,
                         sat_mean_filter=30, val_mean_filter=50)


# def main():
#     wrk_dir = os.path.dirname(__file__)

#     style_image_dir = Path(os.path.join(wrk_dir, 'style_images/train_2'))
#     all_style_images = list(style_image_dir.glob('*.jpg'))

#     collect_results = {'style_image': [], 'H_extrema': [], 'S_extrema': [], 'V_extrema': [],
#                        'H_std': [], 'S_std': [], 'V_std': []}
#     for style_image in tqdm(all_style_images):
#         results_img = get_stat_for_one_style_image(style_image)

#         collect_results['style_image'].append(str(style_image))
#         collect_results['H_extrema'].append(results_img['H_extrema'])
#         collect_results['S_extrema'].append(results_img['S_extrema'])
#         collect_results['V_extrema'].append(results_img['V_extrema'])
#         collect_results['H_std'].append(results_img['H_std'])
#         collect_results['S_std'].append(results_img['S_std'])
#         collect_results['V_std'].append(results_img['V_std'])

#     df = pd.DataFrame.from_dict(collect_results)
#     df['tmp_file_name'] = df['style_image'].apply(lambda x: Path(x).stem).apply(pd.to_numeric)
#     df.sort_values(by='tmp_file_name', inplace=True)
#     df.drop('tmp_file_name', axis=1, inplace=True)
#     output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_hsv.csv')
#     df.to_csv(output_path, index=False)

#     # order by lowest std
#     df['min_std'] = df[['H_std', 'S_std', 'V_std']].min(axis=1)
#     df['mean_std'] = df[['H_std', 'S_std', 'V_std']].mean(axis=1)
#     df.sort_values(by='mean_std', inplace=True)
#     output_path = os.path.join(wrk_dir, 'style_images/train_2/statistics_ordered_by_hsv.csv')
#     df.to_csv(output_path, index=False)

#     # filter images
#     out_dir = style_image_dir / 'filtered_images'
#     out_dir.mkdir(exist_ok=True)
#     filter_images(df, out_dir)

if __name__ == '__main__':
    main()
