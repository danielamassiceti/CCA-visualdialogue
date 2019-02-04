import torch, sys, os, json, inflect, torchvision
import torch.nn as nn
from nltk import word_tokenize
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

nc = inflect.engine()

# user-defined packages
import utils
from corpus import Dictionary
from datasets import VisualDialogDataset

def build_dataset(which_set, opt, dictionary):
    
    if which_set == 'val' or which_set == 'test':
        withoptions = True
    else:
        withoptions = False

    normalize = transforms.Normalize(mean=[0.4711, 0.4475, 0.4080], std=[0.1223, 0.1221, 0.1450]) #visdial 
    transform = transforms.Compose([    transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])
    
    return VisualDialogDataset(which_set, opt, dictionary, withoptions, transform)

def build_vocabulary(opt, log):
    
    sets = [['train'], [False]] # build vocab from just train set, with options=False

    vocabpath = os.path.join(opt.datasetdir, str(opt.datasetversion), 'vocab_visdial_v' + str(opt.datasetversion) + '_' + ''.join(sets[0]) + '.pt')
    
    if os.path.exists(vocabpath):
        log.info('loading ' + opt.dataset + ' dictionary from ' + ''.join(sets[0]) + ' captions and dialogues')
        vocab = torch.load(vocabpath)
    else:
        log.info('building ' + opt.dataset + ' dictionary from ' + ''.join(sets[0]) + ' captions and dialogues... (takes several minutes)')
        
        vocab = build_visdial_dictionary(opt, sets)
        torch.save(vocab, vocabpath)

    return vocab

def build_visdial_dictionary(opt, sets):
    
    vocab = Dictionary()

    special_tokens = ['<PAD>', '<EOS>', '<UNK>']
    for t in special_tokens:
        vocab.add_word(t)
    vocab.set_UNK_ID('<UNK>')
    
    setnames = sets[0]
    withoptions = sets[1]
        
    for s, whichset in enumerate(setnames):
        
        annfile = 'visdial_' + str(opt.datasetversion) + '_' + whichset + '.json'
        
        with open(os.path.join(opt.datasetdir, str(opt.datasetversion), annfile)) as data_file:
            jdata = json.load(data_file)

        questions = jdata['data']['questions']
        answers = jdata['data']['answers']

        for i in jdata['data']['dialogs']:

            dialog = i['dialog'] #get dialog for that image
        
            ctokens = word_tokenize(utils.preprocess(i['caption'], nc)) 
            for ct in ctokens:
                vocab.add_word(ct)
        
            ndialog = len(dialog)
            for d in dialog:
                candidates = d['answer_options']
            
            	# preprocess & tokenize
                q_idx = d['question']
                a_idx = d['answer']
                qtokens = word_tokenize(utils.preprocess(questions[q_idx], nc))
                atokens = word_tokenize(utils.preprocess(answers[a_idx], nc))
                
                allotokens = []
                if withoptions[s]:
                    for op,op_idx in enumerate(candidates):
                        otokens = word_tokenize(utils.preprocess(answers[op_idx], nc))
                    allotokens += otokens
             
                for t in qtokens+atokens+allotokens:
                    vocab.add_word(t)

    vocab.filterbywordfreq(special_tokens, 5) # remove words with freq of < 5

    return vocab

def get_dataloader(set, opt, dictionary, log):

    log.info('loading ' + set + ' data from ' + os.path.join(opt.datasetdir, str(opt.datasetversion)))

    loader = torch.utils.data.DataLoader(
                    build_dataset(set, opt, dictionary),
                    batch_size=opt.batch_size, shuffle=False,
                    num_workers=opt.workers, pin_memory=False)
    
    nelements = len(loader.dataset)
    log.info('loading successful. # ' + set + ' items: ' + str(nelements))
    log.info('-' * 100)

    return loader

def get_features(dictionary, opt, log, set):
 
    loader = get_dataloader(set, opt, dictionary, log)
    
    featuredir = os.path.join(opt.datasetdir, str(opt.datasetversion), 'features')
    img_feature_file = os.path.join(featuredir, set + '_' + opt.imagemodel + '_img_feats.pt')
    cap_feature_file = os.path.join(featuredir, set + '_' + os.path.basename(opt.wordmodel) + '_cap_feats.pt')
    quest_feature_file = os.path.join(featuredir, set + '_' + os.path.basename(opt.wordmodel) + '_quest_feats.pt')
    ans_feature_file = os.path.join(featuredir, set + '_' + os.path.basename(opt.wordmodel) + '_ans_feats.pt')
    
    #load from saved
    if os.path.exists(img_feature_file) and os.path.exists(cap_feature_file) and os.path.exists(quest_feature_file) and os.path.exists(ans_feature_file):         
        
        V1 = torch.load(img_feature_file)
        V2 = torch.load(cap_feature_file)
        V3 = torch.load(quest_feature_file)
        V4 = torch.load(ans_feature_file)

    else: #get features on the fly

        log.info('getting pre-trained features for ' + set + ' images, questions and answers...')

        # build image feature network
        img_model = torchvision.models.__dict__[opt.imagemodel](pretrained=True) # use pre-trained weights
        if 'resnet' in opt.imagemodel:
            img_feat_net = nn.Sequential(*list(img_model.children())[:-1])
        else:
            img_feat_net = nn.ModuleList([img_model.features, nn.Sequential(*list(img_model.classifier.children())[:-1])])
        img_feat_net.eval()
        for p in img_feat_net.parameters():
            p.requires_grad = False
        if opt.gpu>=0:
            img_feat_net.to('cuda:' + str(opt.gpu))
        
        V1 = torch.zeros(len(loader.dataset), 512 if 'resnet' in opt.imagemodel else 4096) # resnet feature dim
        V2 = torch.zeros(len(loader.dataset), opt.emsize) # avg fasttext dim
        V3 = torch.zeros(len(loader.dataset), opt.exchangesperimage, opt.emsize) # avg fasttext dim
        V4 = torch.zeros(len(loader.dataset), opt.exchangesperimage, opt.emsize) # avg fasttext dim
         
        for i, batch in enumerate(loader):

            sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(i+1), str(len(loader)), (i+1)/float(len(loader))*100))
            sys.stdout.flush()

            bsz = batch['img'].size(0)
            batch = utils.send_to_device(batch, opt.gpu)
            
	    # bsz x 512 image features
            img_feat = img_feat_net(batch['img']) if 'resnet' in opt.imagemodel else img_feat_net[1](img_feat_net[0](batch['img']).view(bsz, -1))
            V1[i*loader.batch_size:i*loader.batch_size+bsz] = img_feat.detach().squeeze().cpu()
	
            # bsz x opt.emsize average caption embeddings
            V2[i*loader.batch_size:i*loader.batch_size+bsz] = utils.get_avg_embedding(batch['caption_ids'].unsqueeze(1), batch['caption_length'].unsqueeze(1), dictionary).squeeze(1).cpu()

            # bsz x opt.emsize average question embeddings
            V3[i*loader.batch_size:i*loader.batch_size+bsz] = utils.get_avg_embedding(batch['questions_ids'], batch['questions_length'], dictionary).cpu()
             
            # bsz x opt.emsize average answer embeddings
            V4[i*loader.batch_size:i*loader.batch_size+bsz] = utils.get_avg_embedding(batch['answers_ids'], batch['answers_length'], dictionary).cpu()

        sys.stdout.write("\n")
        os.makedirs(featuredir, exist_ok=True)
        
        img_feat_net.to('cpu')
        torch.save(V1, img_feature_file)
        torch.save(V2, cap_feature_file)
        torch.save(V3, quest_feature_file)
        torch.save(V4, ans_feature_file)
        log.info('-' * 100)

    return loader, {'img': V1, 'caption': V2, 'question': V3, 'answer': V4}

def filter_features(flat_features, input_var, condition_vars):

    # filter features by input_var and condition_vars
    filtered_features = [flat_features[input_var]]
    filtered_features.extend([flat_features[c] for c in condition_vars])
    
    # compute means 
    mus = [f.mean(dim=0).view(-1,1) for f in filtered_features]

    # ensure matching in size
    filtered_features = utils.size_match( filtered_features )

    return filtered_features, mus

