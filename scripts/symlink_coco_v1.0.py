# loads visual dialog .json file from source destination and creates a folder in target destination of symlinked ms-coco images
# to handle v1.0 training images, which are a combination of v0.9 train and validation MS-COCO images

# input args
    # visdialdir = root directory of visdial
    # cocodir = root directory of ms-coco
    # source_version = source version of visdial 
    # target_version = target version of visdial

import argparse
import os, gc
import json
from pprint import pprint

parser = argparse.ArgumentParser(description="Folderize MS-COCO")
parser.add_argument('--visdialdir', dest='visdialdir', help='VisDial root directory')
parser.add_argument('--cocodir', dest='cocodir', help='MS-COCO root directory')
parser.add_argument('--source_version', dest='source_version', type=float, default=0.9, help='VisDial version to copy from')
parser.add_argument('--target_version', dest='target_version', type=float, default=1.0, help='VisDial version to copy to')

opt = parser.parse_args()

if not os.path.exists(os.path.join(opt.visdialdir, str(opt.target_version), 'visdial_' + str(opt.source_version) + '_train.json')) or not os.path.exists(os.path.join(opt.visdialdir, str(opt.target_version), 'visdial_' + str(opt.source_version) + '_val.json')):
    print ('Please place visdial_' + str(opt.source_version) + '_train.json and visdial_' + str(opt.source_version) + '_val.json in your visdial directory!')
    exit()

# set destination folder
destdir = os.path.join(opt.visdialdir, str(opt.target_version), 'train')
if not os.path.exists(os.path.join(destdir, 'cls1')):
    os.makedirs(os.path.join(destdir, 'cls1'))

# load visdial .json data
with open(os.path.join(opt.visdialdir, str(opt.target_version), 'visdial_' + str(opt.target_version) + '_train.json')) as vd_file:
    vddata = json.load(vd_file)

vddata_old = []
with open(os.path.join(opt.visdialdir, str(opt.target_version), 'visdial_' + str(opt.source_version) + '_train.json')) as vd_file1:
    vddata_old.append(json.load(vd_file1))
with open(os.path.join(opt.visdialdir, str(opt.target_version), 'visdial_' + str(opt.source_version) + '_val.json')) as vd_file2:
    vddata_old.append(json.load(vd_file2))

nentries = len(vddata)

print ('Symlinking...')
for s_e_t in vddata_old:
    imgsplit = s_e_t['split']
    for item in s_e_t['data']['dialogs']:
        imgid = item["image_id"]
        cocoimgname = 'COCO_' + imgsplit + '_' + str(imgid).zfill(12) + '.jpg'
        cocoimgpath = os.path.join(opt.cocodir, 'images', imgsplit, cocoimgname)
        vdimgpath = os.path.join(destdir, 'cls1', 'VisualDialog_train_' + str(imgid).zfill(12) + '.jpg')
        os.symlink(cocoimgpath, vdimgpath)
        print (cocoimgpath + '\t ----> \t' + vdimgpath)

print ('Symlinking done!')

del vddata, vddata_old
gc.collect()
