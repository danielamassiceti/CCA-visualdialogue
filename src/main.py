import cv2, os, time, sys, torch, json, pdb, gc
import torch.backends.cudnn as cudnn
import numpy as np

import visdial as dataset
import args, corpus, utils, cca_utils, evals

def main():

    opt = args.getopt()

    # set-up file structures and create logger
    datasetdir = os.path.join(opt.datasetdir, str(opt.datasetversion))
    opt.batch_size == 1 if opt.save_ranks else opt.batch_size
    resultsdir = os.path.join(opt.resultsdir, 'experiment_id' + str(opt.id))
    if not os.path.exists(resultsdir):
        os.system('mkdir -p ' + resultsdir)
    log = utils.generate_logger(opt, resultsdir)
    
    # basic/cuda set-up
    torch.manual_seed(opt.seed)
    if opt.gpu>=0:
        assert opt.gpu <= torch.cuda.device_count()
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.set_device(opt.gpu)
        cudnn.enabled = True
        cudnn.benchmark = True
    else:
        log.warning('on cpu. you should probably use gpus with --gpu=GPU_idx')
 
    
    # print & save arguments
    log.info(opt)
    torch.save(opt, os.path.join(resultsdir, 'exp' + str(opt.id) + '_opt.pt'))
     
    ####################################### build vocab  #####################################
    log.info('-' * 100)
    dictionary = dataset.build_vocabulary(opt, log)
    ntokens = len(dictionary)
    log.info('dictionary loaded successfully! vocabulary size: ' + str(ntokens))
    log.info('-' * 100)

    # get pre-trained word embeddings (from downloaded binary file saved at opt.wordmodel)
    word_vec_file = os.path.join(opt.datasetdir, str(opt.datasetversion), os.path.basename(opt.wordmodel) + '_vocab_vecs.pt')
    dictionary.loadwordmodel(opt.wordmodel, word_vec_file, opt.emsize, log, opt.gpu)
    log.info('-' * 100)
    
    ####################################### load data ##################################### 
 
    train_loader, flat_train_features = dataset.get_features(dictionary, opt, log, 'train') 
    test_loader, flat_test_features = dataset.get_features(dictionary, opt, log, opt.evalset) 

    # filter views by input_vars and condition_vars
    condition_vars = sorted(opt.condition_vars.split('_')) # sorted alphabetically
    train_views, train_mus = dataset.filter_features(flat_train_features, opt.input_vars, condition_vars)
    test_views, _ = dataset.filter_features(flat_test_features, opt.input_vars, condition_vars)

    # do CCA
    lambdas, proj_mtxs = cca_utils.cca(train_views, log, k=opt.k)
    #lambdas, proj_mtxs = cca_utils.cca_mardia(train_views, log, opt.k, opt.eps, opt.r)
    
    # get (train) projections using learned weights
    train_projections = [cca_utils.get_projection(v, mtx, lambdas, opt.p) for (v, mtx) in zip(train_views, proj_mtxs) ]
    proj_train_mus = [ proj.mean(dim=0).view(-1,1) for proj in train_projections ]

    if len(condition_vars) == 2: # three-view CCA
 
        # select answer & question proj_mtxs (rather than image)
        qa_proj_mtxs = [proj_mtxs[0], proj_mtxs[2]]
        qa_train_projections = [train_projections[0], train_projections[2]]
        qa_proj_train_mus = [proj_train_mus[0], proj_train_mus[2]]
        
        test_meters = evals.candidate_answers_recall(test_loader, lambdas, qa_proj_mtxs, qa_train_projections, 
                qa_proj_train_mus, dictionary, opt, log, opt.evalset, train_loader)
        
    elif len(condition_vars) == 1: # two-view CCA 
        test_meters = evals.candidate_answers_recall(test_loader, lambdas, proj_mtxs, train_projections, 
                proj_train_mus, dictionary, opt, log, opt.evalset, train_loader)
            
    else:
        print ('cannot handle this CCA architecture - check --input_vars and --condition_vars!')
        return
    
    log.info('-' * 100) 
    log.info(opt)
    log.info('-' * 100)
    s = utils.stringify_meters(test_meters)
    log.info(s)

    # saves ranking performance to text file with p,k parameters 
    resultsdir = os.path.join(opt.resultsdir, 'experiment_id' + str(opt.id))
    fh = open(os.path.join(resultsdir, 'p_k_gridsearch.txt'), 'a')
    fh.write('k={k}\tp={p}\tranks: {ranks}\r\n'.format(k=str(opt.k), p=str(opt.p), ranks=s))
    fh.close()

if __name__ == '__main__':
    main()

