# Check the style images to exclude the images that are very monochrome with little contrast
import os
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageStat


def get_stat_for_one_style_image(style_image_path):
    style_img = Image.open(str(style_image_path)).convert('RGB')
    stats = ImageStat.Stat(style_img)

    results = {}
    for band, name in enumerate(style_img.getbands()): 
        results['{}_extrema'.format(name)] = stats.extrema[band]
        results['{}_std'.format(name)] = stats.stddev[band]
        # print(f'Band: {name}, min/max: {stats.extrema[band]}, stddev: {stats.stddev[band]}')

    return results


def filter_images(stats_df, out_dir, filter_thresholds=(25, 100)):
    filtered_df = stats_df.loc[(stats_df['mean_std'] < filter_thresholds[0]) | (stats_df['mean_std'] > filter_thresholds[1])]
    to_filter_out = filtered_df['style_image'].values.tolist()
    for img in to_filter_out:
        img = Path(img)
        out_file = out_dir / img.name
        print('Moving image {} from {} to {}'.format(img.name, str(img.parent), str(out_dir)))
        shutil.move(str(img), str(out_file))


def main():
    wrk_dir = os.path.dirname(__file__)

    style_image_dir = Path(os.path.join(wrk_dir, 'style_images/train_2'))
    all_style_images = list(style_image_dir.glob('*.jpg'))

    collect_results = {'style_image': [], 'R_extrema': [], 'G_extrema': [], 'B_extrema': [],
                       'R_std': [], 'G_std': [], 'B_std': []}
    for style_image in tqdm(all_style_images):
        results_img = get_stat_for_one_style_image(style_image)

        collect_results['style_image'].append(str(style_image))
        collect_results['R_extrema'].append(results_img['R_extrema'])
        collect_results['G_extrema'].append(results_img['G_extrema'])
        collect_results['B_extrema'].append(results_img['B_extrema'])
        collect_results['R_std'].append(results_img['R_std'])
        collect_results['G_std'].append(results_img['G_std'])
        collect_results['B_std'].append(results_img['B_std'])

    df = pd.DataFrame.from_dict(collect_results)
    df['tmp_file_name'] = df['style_image'].apply(lambda x: Path(x).stem).apply(pd.to_numeric)
    df.sort_values(by='tmp_file_name', inplace=True)
    df.drop('tmp_file_name', axis=1, inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/test_images/statistics.csv')
    df.to_csv(output_path, index=False)

    # order by lowest std
    df['min_std'] = df[['R_std', 'G_std', 'B_std']].min(axis=1)
    df['mean_std'] = df[['R_std', 'G_std', 'B_std']].mean(axis=1)
    df.sort_values(by='mean_std', inplace=True)
    output_path = os.path.join(wrk_dir, 'style_images/test_images/statistics_ordered.csv')
    df.to_csv(output_path, index=False)

    # filter images
    out_dir = style_image_dir / 'filtered_images'
    out_dir.mkdir(exist_ok=True)
    filter_images(df, out_dir)

if __name__ == '__main__':
    main()
