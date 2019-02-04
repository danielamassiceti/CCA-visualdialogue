# loads visual dialog v0.9 .json file and creates a folder of symlinked ms-coco images

# input args
    # visdialdir = root directory of visdial
    # cocodir = root directory of ms-coco
    # visdialversion = which version of visdial 
    # set = train, val

import argparse
import os, gc
import json
from pprint import pprint

parser = argparse.ArgumentParser(description="Folderize MS-COCO")
parser.add_argument('--visdialdir', dest='visdialdir', help='VisDial root directory')
parser.add_argument('--cocodir', dest='cocodir', help='MS-COCO root directory')
parser.add_argument('--visdialversion', dest='visdialversion', type=float, default=0.9, help='VisDial version')
parser.add_argument('--set', dest='set', default='val', help='Options = train, val')

opt = parser.parse_args()

# set destination folder
destdir = os.path.join(opt.visdialdir, str(opt.visdialversion), opt.set)
if not os.path.exists(os.path.join(destdir, 'cls1')):
    os.makedirs(os.path.join(destdir, 'cls1'))

# load visdial .json data
with open(os.path.join(opt.visdialdir, str(opt.visdialversion), 'visdial_' + str(opt.visdialversion) + '_' + opt.set + '.json')) as vd_file:
    vddata = json.load(vd_file)

nentries = len(vddata)
imgsplit = vddata['split']

print ('Symlinking...')
for item in vddata['data']['dialogs']:
    imgid = item["image_id"]
    cocoimgname = 'COCO_' + imgsplit + '_' + str(imgid).zfill(12) + '.jpg'
    cocoimgpath = os.path.join(opt.cocodir, 'images', imgsplit, cocoimgname)
    os.symlink(cocoimgpath, os.path.join(destdir, 'cls1', cocoimgname))
    print (cocoimgpath + '\t ----> \t' + os.path.join(destdir, 'cls1', cocoimgname))

print ('Symlinking done!')

del vddata
gc.collect()
