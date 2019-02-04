import os, logging, json, re, math, cv2, torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata

def send_to_device(input_batch, device):

    if device>=0:
        if isinstance(input_batch, dict):
            for k, tensor in input_batch.items():
                if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                    input_batch[k] = tensor.to('cuda:' + str(device))
        elif isinstance(input_batch, list):
            for k, tensor in enumerate(input_batch):
                if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                    input_batch[k] = tensor.to('cuda:' + str(device))
        elif isinstance(input_batch, torch.Tensor):
            if not input_batch.is_cuda:
                input_batch = input_batch.to('cuda:' + str(device))
    
    return input_batch

def load_glove_embeddings(filename):
    
    with open(filename, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = np.asarray([float(x) for x in vals[1:]])

    return vectors

def size_match(views):

    # assumes views are either 2- or 3-dimensional
    # must get all views to (-1, v_i_emb) with correct pairing if views differ in dimensionality

    dims = [v.dim() for v in views]
    max_dim = max(dims)
    max_view = views[dims.index(max_dim)]

    for i,v in enumerate(views):
        if v.dim() == 2:
            if v.dim() < max_dim:
                v = v.unsqueeze(1).expand(v.size(0), max_view.size(1), v.size(1)).contiguous()
            else:
                v = v.unsqueeze(1)

        views[i] = v.view(-1, v.size(2))
    
    return views

def process_ranks_to_save(ranks_to_save, img_names, ranks, round_ids, which_set):

    for i, img_ranks in enumerate(ranks):
        if which_set == 'test':
            round_id = round_ids[ torch.ne(round_ids, -1) ].item()
            ranks_to_save.append({
                    'image_id': int(img_names[i].split('.')[0][-12:]),
                    'round_id': round_id + 1, 
                    'ranks': img_ranks[round_id].cpu().int().numpy().tolist()
                    })
        else: # train/val set
            print (img_ranks)
            for ii, img_rank in enumerate(img_ranks):
                print (ii)
                ranks_to_save.append({
                    'image_id': int(img_names[i].split('.')[0][-12:]),
                    'round_id': ii + 1, 
                    'ranks': img_rank.cpu().int().numpy().tolist()
                    })

    
    return ranks_to_save

def process_ranks_for_meters(meters, gt_ranks, sorted_corrs=None, on_the_fly=False):
   
    if not on_the_fly: # does not compute ranks when on_the_fly == True since gtidxs are meaningless here
        meters['mrank'].update( list(gt_ranks.cpu().view(-1).numpy()) )
   
        # recall@k
        kvals = [int(k[k.index('@')+1:]) for k,v in meters.items() if 'recall' in k]
        for k in kvals:
            intopk = torch.le(gt_ranks, k).float().mul_(100) # bsz x nexchanges
            meters['recall@' + str(k)].update( list(intopk.cpu().view(-1).numpy()) )
            meters['rrank'].update( list( (1.0 / gt_ranks).cpu().view(-1).numpy()) )
    
    if isinstance(sorted_corrs, torch.Tensor): # if --threshold 
        np_sorted_corrs = sorted_corrs.cpu().numpy()
        # computes isodata threshold on ranks and variance
        for bb in range(gt_ranks.size(0)):
            for nn in range(gt_ranks.size(1)):
                # computes threshold on correlations 
                threshold = threshold_isodata(np_sorted_corrs[bb][nn], nbins=5)
            
                gt_rank = gt_ranks[bb][nn].long()
                meters['ge_thresh'].update( (sorted_corrs[bb][nn][gt_rank-1] >= threshold).float().mul_(100).item() )
     
                ge_threshold_idxs = torch.ge(sorted_corrs[bb][nn], threshold)
                if ge_threshold_idxs.float().sum() > 1:
                    meters['ge_thresh_var'].update( torch.var( sorted_corrs[bb][nn][ ge_threshold_idxs ]).item() )

    return meters
    
def stringify_meters(meters):
    s = 'meters: '
    for k,v in meters.items():
        s += '\t{name}: {meter: 6.4f}'.format(name=k, meter=v.avg)
    if meters['mrank'].count > 0:
        s += '\tmedian: {median: 6.4f}'.format(median=np.median(meters['mrank'].values))
    return s + '\n'

def log_iteration_stats(log, meters, iteration, all_iterations):
    s = ''
    s += 'Batch [{:d}/{:d}]'.format(iteration, all_iterations)
    for k,v in meters.items():
        s += '\t{name}: {meter.val: 6.4f} ({meter.avg: 6.4f})'.format(name=k, meter=v)

    log.info(s)

def save_meters(meters, save_path):

    for k,v in meters.items():
        filename = os.path.join(save_path + '_' + k + '.meter')
        if not os.path.exists(filename):
            torch.save(v, filename)

def generate_logger(opt, resultsdir, mode=0):

    # mode 0 (train) mode 1 (evaluate)
    if mode == 1:
        LOG_FILENAME = os.path.join(resultsdir, 'exp' + str(opt.id) + '.testlog')
    else:
        LOG_FILENAME = os.path.join(resultsdir, 'exp' + str(opt.id) + '.log')
    FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
    FORMATCONSOLE = "%(message)s"
    logger = logging.getLogger(LOG_FILENAME)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILENAME)
    formatter = logging.Formatter(FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    fhconsole = logging.StreamHandler()
    formatterconsole = logging.Formatter(FORMATCONSOLE)
    fhconsole.setFormatter(formatterconsole)
    logger.addHandler(fhconsole)
    return logger

# convert digits to words
def digitreplacer(digit, numconverter):
    return numconverter.number_to_words((digit)).replace("-"," ")

# gets token list to be of length opt.seqlen using EOS and PAD tokens
# if questions/answers are shorter than opt.seqlen, then pad with PAD
# if questions/answers are longer than opt.seqlen, then chop-chop!
def pad_to_max_length(tokenlist, seqlen, dictionary, padtoken='<PAD>', eostoken='<EOS>', unktoken='<UNK>'):

    PAD_ID = dictionary.word2idx[padtoken]
    EOS_ID = dictionary.word2idx[eostoken]
    UNK_ID = dictionary.word2idx[unktoken]

    tokenlist = [dictionary.word2idx.get(t, UNK_ID) for t in tokenlist]
    t_len = len(tokenlist)
    
    suffix = [EOS_ID]
    if t_len < seqlen:
        suffix.extend( [ PAD_ID for x in range(seqlen - t_len - 1) ] ) # add EOS PAD PAD PAD ...
        return torch.LongTensor(tokenlist+suffix), t_len+1
    else:
        return torch.LongTensor(tokenlist[0:seqlen-1]+suffix), seqlen

def preprocess(sentence, numconverter):

    # remove all apostrophes
    sentence = sentence.replace("'", "")    
    #sentence = ' '.join([word for word in sentence.split() if word not in cached_stop_words])
    sentence = re.sub(r'\d+', lambda x: digitreplacer(x.group(), numconverter), sentence).lower()
    
    return sentence
 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if not isinstance(val, list):
            val = [val]
        self.values.extend(val)
        self.val = val[-1] 
        self.sum += sum(val)
        self.count += len(val)
        self.avg = self.sum / float(self.count)

def get_avg_embedding(idx_batch, idx_lens, dictionary):

    word_embeddings = get_word_embeddings(idx_batch, dictionary)

    return word_embeddings.sum(dim=2).div_(idx_lens.float().unsqueeze(2))

def get_word_embeddings(idx_batch, dictionary):

    return dictionary.word_embs.index_select(0, idx_batch.view(-1)).view(idx_batch.size(0), idx_batch.size(1), idx_batch.size(2), -1)

