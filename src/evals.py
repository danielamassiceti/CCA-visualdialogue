import cv2, torch, os, json, pdb
import utils, cca_utils
import random

def nearest_neighbour_baselines(test_loader, dictionary, opt, log, set, train_views, test_views):

    kvals, meters = [1, 5, 10], {}
    meters['mrank'] = utils.AverageMeter()
    for k in kvals:
        meters['recall@' + str(k)] = utils.AverageMeter()
    meters['rrank'] = utils.AverageMeter()
    
    log.info('computing nearest-neighbour baseline with ' + str(opt.condition_vars))

    k=100
    kp=20

    emb_train_answers = utils.send_to_device(train_views[0], opt.gpu)
    emb_train_questions = utils.send_to_device(train_views[-1], opt.gpu)
    if len(train_views) == 3:
        emb_train_images = utils.send_to_device(train_views[1], opt.gpu)
        emb_test_images = utils.send_to_device(test_views[1], opt.gpu)
        img_feat_size = emb_train_images.size(1)
    else:
        emb_train_images = None
    
    N = emb_train_answers.size(0)

    for i, batch in enumerate(test_loader):

        # current batch size
        bsz = batch['img'].size(0)
        batch = utils.send_to_device(batch, opt.gpu)
        
        # averaged word vectors for question
        emb_question = utils.get_avg_embedding(batch['questions_ids'], batch['questions_length'], dictionary) # bsz x nexchanges x 300

        # averaged word vectors for answer candidates
        emb_opts = utils.get_avg_embedding(batch['answer_options_ids'].view(-1, 100, opt.seqlen), batch['answer_options_length'].view(-1, 100), dictionary)
      
        dists = torch.norm( emb_train_questions.unsqueeze(1).unsqueeze(1).expand(N, bsz, opt.exchangesperimage, opt.emsize) 
                - emb_question.unsqueeze(0).expand(N, bsz, opt.exchangesperimage, opt.emsize), dim=3, p=2) # N x bsz x opt.exchangesperimage
        topk_train_question_idxs = torch.topk(dists, k=k, dim=0, largest=False)[1] # k x bsz x opt.exchangesperimage
        topk_train_emb_answers = emb_train_answers.index_select(0, topk_train_question_idxs.view(-1)).view(k, bsz, opt.exchangesperimage, opt.emsize)
        mean_train_answer = topk_train_emb_answers.mean(dim=0)

        if len(train_views) == 3: # further filter ids with image features
            test_emb_images = emb_test_images[i*opt.exchangesperimage*opt.batch_size : i*opt.exchangesperimage*opt.batch_size+bsz*opt.exchangesperimage]
            test_emb_images = test_emb_images.view(bsz, opt.exchangesperimage, img_feat_size)
            dists = torch.norm( emb_train_images.index_select(0, topk_train_question_idxs.view(-1)).view(k, bsz, opt.exchangesperimage, img_feat_size) - test_emb_images.unsqueeze(0).expand(k, bsz, opt.exchangesperimage,
                img_feat_size), p=2, dim=3)
            topkp_train_question_idxs = torch.topk(dists, k=kp, dim=0, largest=False)[1] # kp x bsz x opt.exchangesperimage

            topkp_train_question_idxs = topkp_train_question_idxs.unsqueeze(-1).expand(kp, bsz, opt.exchangesperimage, opt.emsize)
            mean_train_answer = topk_train_emb_answers.gather(0, topkp_train_question_idxs).view(kp, bsz, opt.exchangesperimage, opt.emsize).mean(dim=0)

        dists = torch.norm( emb_opts - mean_train_answer.unsqueeze(2).expand(bsz, opt.exchangesperimage, 100, opt.emsize), p=2, dim=3) # bsz x opt.exchangesperimage x 100
        sorted_dists, indices_dists = torch.sort(dists, dim=2, descending=False)
        
        # compute ranks 
        ranks = torch.zeros(sorted_dists.size()).type_as(sorted_dists)
        ranks.scatter_(2, indices_dists, torch.arange(1,101).type_as(sorted_dists).view(1,1,100).expand_as(sorted_dists)) # bsz x nexchanges
            
        gt_ranks = ranks.gather(2, batch['gtidxs'].unsqueeze(-1))
        meters = utils.process_ranks_for_meters(meters, gt_ranks, sorted_corrs if opt.threshold else None, opt.on_the_fly) 
        utils.log_iteration_stats(log, meters, i+1, len(test_loader)) 

    return meters

# computes ranking performance on candidate set of answers
def candidate_answers_recall(test_loader, lambdas, proj_mtxs, train_projections, proj_train_mus, dictionary, opt, log, set, train_loader=None):

    log.info('computing ranks with on-the-fly candidates = ' + str(opt.on_the_fly))
   
    torch.autograd.set_grad_enabled(False)

    # set up meters and buffers
    meters, ranks_to_save = {}, []
    if opt.on_the_fly:
        # create buffers for projected candidates
        topk_idx_buffer = torch.zeros(test_loader.batch_size, opt.exchangesperimage, 100).long()
        topk_idx_buffer = utils.send_to_device(topk_idx_buffer, opt.gpu)

        proj_opts_buffer = torch.zeros(test_loader.batch_size, opt.exchangesperimage, 100, opt.k)
        proj_opts_buffer = utils.send_to_device(proj_opts_buffer, opt.gpu)

        # mean centred train question projections
        proj_q_train = cca_utils.mean_center(train_projections[1], proj_train_mus[1])
        proj_q_train = utils.send_to_device(proj_q_train, opt.gpu)
        train_projections[0] = utils.send_to_device(train_projections[0], opt.gpu)
    else: # only compute ranks when not on_the_fly since gtidxs are meaningless
        kvals = [1, 5, 10]
        meters['mrank'] = utils.AverageMeter()
        for k in kvals:
            meters['recall@' + str(k)] = utils.AverageMeter()
        meters['rrank'] = utils.AverageMeter()
    
    if opt.threshold:
        meters['ge_thresh'] = utils.AverageMeter()
        meters['ge_thresh_var'] = utils.AverageMeter()
    
    proj_mtxs = utils.send_to_device(proj_mtxs, opt.gpu)
    proj_train_mus = utils.send_to_device(proj_train_mus, opt.gpu)
    lambdas = utils.send_to_device(lambdas, opt.gpu)
    
    b1 = proj_mtxs[0]
    b2 = proj_mtxs[1]
        
    N = len(test_loader.dataset)

    for i, batch in enumerate(test_loader):

        # current batch size
        bsz = batch['img'].size(0)
        batch = utils.send_to_device(batch, opt.gpu)

        # averaged word vectors for question
        emb_question = utils.get_avg_embedding(batch['questions_ids'], batch['questions_length'], dictionary) # bsz x nexchanges x 300

        # project question to joint space using b2
        proj_q = cca_utils.get_projection(emb_question.view(-1, opt.k), b2, lambdas, opt.p)
        proj_q = cca_utils.mean_center(proj_q, proj_train_mus[1]) # center by projected train question mean
        proj_q = proj_q.view(bsz, opt.exchangesperimage, 1, opt.k)
        
        # compute candidate answer set
        if opt.on_the_fly:
            topk_train_question_idxs = topk_idx_buffer[0:bsz].view(-1, 100).fill_(0)
            proj_opts = proj_opts_buffer[0:bsz].view(-1, 100, opt.k).fill_(0)
            for q_i, q in enumerate(proj_q.view(-1, opt.k)): # flatten bsz and opt.exchangesperimage
                # get top-k questions from train set
                topk_train_question_idxs[q_i] = cca_utils.topk_corr_distance(proj_q_train, q.unsqueeze(0), k=100)[1] # k indices
                # get their corresponding answers projections
                proj_opts[q_i] = train_projections[0].index_select(0, topk_train_question_idxs[q_i])
            topk_train_question_idxs = topk_train_question_idxs.view(bsz, opt.exchangesperimage, 100)
            proj_opts = proj_opts.view(bsz, opt.exchangesperimage, 100, opt.k) 
        else:
            emb_opts = utils.get_avg_embedding(batch['answer_options_ids'].view(-1, 100, opt.seqlen), batch['answer_options_length'].view(-1, 100), dictionary)
            emb_opts = emb_opts.view(bsz, opt.exchangesperimage, 100, emb_opts.size(-1)) # bsz x nexchanges x 100 x opt.k
            
            # project answer candidates to joint space using b1
            proj_opts = cca_utils.get_projection(emb_opts.view(-1, opt.k), b1, lambdas, opt.p)
            proj_opts = cca_utils.mean_center(proj_opts, proj_train_mus[0]) # center by projected train answer mean
            proj_opts = proj_opts.view(bsz, opt.exchangesperimage, 100, opt.k) 

        # compute (sorted) correlation between 100 candidates & 1 test question
        denom = torch.norm(proj_opts, p=2, dim=3) * torch.norm(proj_q.expand_as(proj_opts), p=2, dim=3)
        corr = torch.matmul(proj_opts, proj_q.transpose(2,3)).squeeze(-1).div_(denom) # bsz x nexchanges x 100
        sorted_corrs, indices = torch.sort(corr, dim=2, descending=True) # indices: bsz x nexchanges x 100

        # compute ranks 
        ranks = torch.zeros(sorted_corrs.size()).type_as(sorted_corrs)
        ranks.scatter_(2, indices, torch.arange(1,101).type_as(sorted_corrs).view(1,1,100).expand_as(sorted_corrs)) # bsz x nexchanges
        if opt.save_ranks and not opt.on_the_fly:
            ranks_to_save = utils.process_ranks_to_save(ranks_to_save, batch['img_name'], ranks, batch['gtidxs'], set)
        
        if not set == 'test':
            gt_ranks = ranks.gather(2, batch['gtidxs'].unsqueeze(-1))
            meters = utils.process_ranks_for_meters(meters, gt_ranks, sorted_corrs if opt.threshold else None, opt.on_the_fly) 
            utils.log_iteration_stats(log, meters, i+1, len(test_loader)) 
        
        # interactive mode
        if opt.interactive:
            randint = random.randint(0, bsz-1)
            print ('Image: {}'.format(batch['img_name'][randint]))
            for ex in range(opt.exchangesperimage):
                worded_q = test_loader.dataset.all_questions[batch['questions'][randint][ex]]
                worded_gt_a = test_loader.dataset.all_answers[batch['answers'][randint][ex]]
                print ('Question #{:d}/{:d}: {}'.format(ex+1, opt.exchangesperimage, worded_q))
                print ('Ground-truth answer #{:d}/{:d}: {}'.format(ex+1, opt.exchangesperimage, worded_gt_a))
                print ('Candidate answers (on-the-fly=' + str(opt.on_the_fly) + '):')
                if opt.on_the_fly:
                    idxs = zip(topk_train_question_idxs[randint][ex] // opt.exchangesperimage, topk_train_question_idxs[randint][ex] % opt.exchangesperimage)
                    candidates = [train_loader.dataset[idx]['answers'][exchange_idx] for idx, exchange_idx in idxs]
                    worded_candidates = [train_loader.dataset.all_answers[c] for c in candidates]
                else:
                    candidates = batch['answer_options'][randint][ex][ indices[randint][ex] ]
                    worded_candidates = [ test_loader.dataset.all_answers[c] for c in candidates ]
                print (worded_candidates)
                input()

    # set up logging mechanisms
    resultsdir = os.path.join(opt.resultsdir, 'experiment_id' + str(opt.id))
    save_path = os.path.join(resultsdir, 'exp' + str(opt.id) + '_' + set + '_' + opt.input_vars + '_' + opt.condition_vars + '_k_' + str(opt.k) + '_p_' + str(opt.p))
    utils.save_meters(meters, save_path)
    if opt.save_ranks:
        with open(save_path + '_ranks.json', 'w') as outfile:
            json.dump(ranks_to_save, outfile)
            log.info('Ranks saved to ' + save_path + '_ranks.json')
 
    torch.autograd.set_grad_enabled(True)

    return meters
