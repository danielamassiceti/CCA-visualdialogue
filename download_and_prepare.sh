# script to download & prepare Visual Dialogue (v1.0) dataset

if [ $# -ne 2 ]; then
  DATADIR=$PWD/data;
  mkdir -p $DATADIR
elif [ "$1" == "-d" ]; then
  DATADIR=$2;
fi

echo Saving data to... $DATADIR

# download FastText pre-trained word vectors
mkdir -p $DATADIR/wordembeddings
wget 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'
unzip wiki.en.zip -d $DATADIR/wordembeddings
rm wiki.en.zip
mv $DATADIR/wordembeddings/wiki.en.bin $DATADIR/wordembeddings/fasttext.wiki.en.bin
mv $DATADIR/wordembeddings/wiki.en.vec $DATADIR/wordembeddings/fasttext.wiki.en.vec

# if you already have COCO images, set COCODIR appropriately and comment out the COCO downloads below
COCODIR=$DATADIR/mscoco
mkdir -p $COCODIR/images
echo Saving MS COCO images to... $COCODIR

# Download COCO train2014 images
wget 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
unzip train2014.zip -d $COCODIR/images
rm train2014.zip

# Download COCO val2014 images
wget 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
unzip val2014.zip -d $COCODIR/images
rm val2014.zip

# Download and prepare Visual Dialogue v1.0 
VDDIR=$DATADIR/visdial
# cls1 is a dummy subfolder required for PyTorch's dataloaders
mkdir -p $VDDIR/1.0
echo 'Saving Visual Dialog (v1.0) to...' $VDDIR/1.0

# Download Visual Dialog datasets (v0.9 is required, since v1.0 training images are combination of image IDs)
wget 'https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_train.zip'
unzip visdial_0.9_train.zip -d $VDDIR/1.0
rm visdial_0.9_train.zip
wget 'https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_val.zip'
unzip visdial_0.9_val.zip -d $VDDIR/1.0
rm visdial_0.9_val.zip

wget -O visdial_1.0_train.zip 'https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=1'
unzip visdial_1.0_train.zip -d $VDDIR/1.0
rm visdial_1.0_train.zip
wget -O visdial_1.0_val.zip 'https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=1'
unzip visdial_1.0_val.zip -d $VDDIR/1.0
rm visdial_1.0_val.zip
wget -O visdial_1.0_test.zip 'https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip?dl=1'
unzip visdial_1.0_test.zip -d $VDDIR/1.0
rm visdial_1.0_test.zip

# Symlink COCO images into $VDDIR/1.0/train/cls1
# v1.0 training images are a combination of v0.9 train and validation images
mkdir -p $VDDIR/1.0/train/cls1
python scripts/symlink_coco_v1.0.py --cocodir $COCODIR --visdialdir $VDDIR

# Download and copy COCO images into $VDDIR/1.0/{val, test}/cls1, respectively
mkdir -p $VDDIR/1.0/val/cls1
wget -O VisualDialog_val2018.zip 'https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1'
unzip VisualDialog_val2018.zip 
mv VisualDialog_val2018/* $VDDIR/1.0/val/cls1/
rm -rf VisualDialog_val2018
rm VisualDialog_val2018.zip

mkdir -p $VDDIR/1.0/test/cls1
wget -O VisualDialog_test2018.zip 'https://www.dropbox.com/s/mwlrg31hx0430mt/VisualDialog_test2018.zip?dl=1'
unzip VisualDialog_test2018.zip 
mv VisualDialog_test2018/* $VDDIR/1.0/test/cls1/
rm -rf VisualDialog_test2018
rm VisualDialog_test2018.zip

# Download and prepare Visual Dialogue v0.9
mkdir -p $VDDIR/0.9
echo 'Saving Visual Dialog (v0.9) to...' $VDDIR/0.9

# Moving v0.9 .jsons to $VDDIR/0.9 since already downloaded
mv $VDDIR/1.0/visdial_0.9_train.json $VDDIR/0.9/visdial_0.9_train.json
mv $VDDIR/1.0/visdial_0.9_val.json $VDDIR/0.9/visdial_0.9_val.json

# Symlink COCO images into $VDDIR/0.9/{train, val}/cls1
mkdir -p $VDDIR/0.9/train/cls1
python scripts/symlink_coco_v0.9.py --cocodir $COCODIR --visdialdir $VDDIR --set train
mkdir -p $VDDIR/0.9/val/cls1
python scripts/symlink_coco_v0.9.py --cocodir $COCODIR --visdialdir $VDDIR --set val
