#!/bin/bash

source activate pytorchie_v1.0
python src/main.py  --datasetdir data/visdial --datasetversion 0.9 --resultsdir results \
                    --imagemodel resnet34 --wordmodel data/wordembeddings/fasttext.wiki.en.bin \
                    --input_vars answer --condition_vars question --k 300 --p 1.0 \
                    --id 1 --batch_size 256 --evalset val --gpu 0 \
