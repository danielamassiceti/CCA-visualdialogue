import argparse

def getopt():

    parser = argparse.ArgumentParser(description='Canonical Correlation Analysis for Visual Dialogue')
    parser.add_argument('--workers', '-j', dest='workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int, help='GPU ID (-1=cpu, 0=gpu_idx0, ...)')
    
    # data
    parser.add_argument('--dataset', dest='dataset', default='visdial', choices=['visdial'], help='dataset to load')
    parser.add_argument('--datasetdir', dest='datasetdir', default='data/visdial', help='VisDial root directory')
    parser.add_argument('--resultsdir', dest='resultsdir', default='results', help='results directory (default: ./results)')
    parser.add_argument('--datasetversion', dest='datasetversion', default=0.9, choices=[0.9, 1.0], type=float, help='dataset version (default: 0.9)')
    parser.add_argument('--wordmodel', dest='wordmodel', default='data/wordembeddings/fasttext.wiki.en.bin', help='pre-trained word embeddings model')
    parser.add_argument('--evalset', dest='evalset', default='val', choices=['val', 'test'], help='evaluation set (val for v0.9, test for v1.0')
    parser.add_argument('--batch_size', dest='batch_size', default=256, type=int, help='batch size')

    # main model
    parser.add_argument('--input_vars', dest='input_vars', default='answer', choices=['answer'], help='input variables to generate')
    parser.add_argument('--condition_vars', dest='condition_vars', default='question', choices=['question', 'img_question'], help='variables to condition input on')

    # image model settings
    parser.add_argument('--imagemodel', dest='imagemodel', default='resnet34', choices=['resnet18', 'resnet34', 'vgg16', 'vgg19'], help='image model architecture (default: resnet34)')
    
    # dialogue model
    parser.add_argument('--emsize', '-em', dest='emsize', default=300, type=int, help='size of word embeddings')
    parser.add_argument('--seqlen', dest='seqlen', default=16, type=int, help='maximum sequence length')
    parser.add_argument('--exchanges', '-epi', dest='exchangesperimage', default=10, type=int, help='dialog exchanges per image (default=10 for VisDial)')
    
    # experiment                                                                                                             
    parser.add_argument('--id', dest='id', default=1, type=int, help='id string of experiment')
    parser.add_argument('--seed', '--s', dest='seed', default=12345, type=int, help='initial random seed')

    # evaluation
    parser.add_argument('--save_ranks', dest='save_ranks', action='store_true', help='save ranks for upload to evaluation server')
    parser.add_argument('--on_the_fly', dest='on_the_fly', action='store_true', help='compute candidate answers on-the-fly from training set')
    parser.add_argument('--interactive', dest='interactive', action='store_true', help='print out ranked candidate answers')
    parser.add_argument('--threshold', dest='threshold', action='store_true', help='compute threshold analysis')

    #CCA settings
    parser.add_argument('--p', type=float, metavar='D', default=1.0, help='exponent for eigenvalue in CCA')
    parser.add_argument('--k', type=int, metavar='N', default=300, help='joint projection dimensionality in CCA')

    # settings for CCA Mardia et al. only
    parser.add_argument('--eps', type=float, default=1e-12, help='exponent for eigenvalue in first-tier CCA')
    parser.add_argument('--r', type=float, default=1e-4, help='exponent for eigenvalue in first-tier CCA')
    
    return parser.parse_args()
