import argparse
import os, sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scene', required=True, help='path to sens file to read')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.add_argument('--export_label_lift', dest='export_label_lift', action='store_true')
parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True, export_label_lift=True)

opt = parser.parse_args()
print(opt)


def main():
  file_dir = 'scans/{}/{}.sens'.format(opt.scene, opt.scene)
  output_path = 'scannet/{}'.format(opt.scene)
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  if opt.export_label_lift:
    import zipfile
    with zipfile.ZipFile('scans/{}/{}_2d-label-filt.zip'.format(opt.scene, opt.scene), 'r') as zip_ref:
        zip_ref.extractall(output_path)

  # load the data
  sys.stdout.write('loading %s...' % file_dir)
  sd = SensorData(file_dir)
  sys.stdout.write('loaded!\n')
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(output_path, 'depth'))
  if opt.export_color_images:
    sd.export_color_images(os.path.join(output_path, 'color'))
  if opt.export_poses:
    sd.export_poses(os.path.join(output_path, 'pose'))
  if opt.export_intrinsics:
    sd.export_intrinsics(os.path.join(output_path, 'intrinsic'))




if __name__ == '__main__':
    main()