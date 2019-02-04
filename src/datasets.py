import os, torch, inflect, json, sys
from nltk import word_tokenize
from PIL import Image
from torch.utils.data.dataset import Dataset

import utils
nc = inflect.engine()

class VisualDialogDataset(Dataset):
    def __init__(self, which_set, opt, dictionary, withoptions=False, transform=None):

        self.imgdir = os.path.join(opt.datasetdir, str(opt.datasetversion), which_set, 'cls1')
        self.D = opt.exchangesperimage
        self.S = opt.seqlen
        self.withoptions = withoptions
        self.transform = transform
       
        savedir = os.path.join(opt.datasetdir, str(opt.datasetversion), which_set + '_processed_S' + str(self.S) + '_D' + str(self.D) + '_wo' + str(self.withoptions))
        
        if os.path.exists(savedir):
            self.images = torch.load(os.path.join(savedir, 'images.pt'))
            self.captions = torch.load(os.path.join(savedir, 'captions.pt'))
            self.caption_ids = torch.load(os.path.join(savedir, 'caption_ids.pt'))
            self.caption_lengths = torch.load(os.path.join(savedir, 'caption_lengths.pt'))
            self.all_questions = torch.load(os.path.join(savedir, 'all_questions.pt'))
            self.questions = torch.load(os.path.join(savedir, 'questions.pt'))
            self.question_ids = torch.load(os.path.join(savedir, 'question_ids.pt'))
            self.question_lengths = torch.load(os.path.join(savedir, 'question_lengths.pt'))
            self.all_answers = torch.load(os.path.join(savedir, 'all_answers.pt'))
            self.answers = torch.load(os.path.join(savedir, 'answers.pt'))
            self.answer_ids = torch.load(os.path.join(savedir, 'answer_ids.pt'))
            self.answer_lengths = torch.load(os.path.join(savedir, 'answer_lengths.pt'))
            if self.withoptions:
                self.gtidxs = torch.load(os.path.join(savedir, 'gtidxs.pt'))
                self.answer_options = torch.load(os.path.join(savedir, 'answer_options.pt'))
                self.answer_options_ids = torch.load(os.path.join(savedir, 'answer_options_ids.pt'))
                self.answer_options_lengths = torch.load(os.path.join(savedir, 'answer_options_lengths.pt')) 

            self.N = len(self.images)
        
        else:
            print ('preprocessing dialogues and saving to ' + savedir)
            os.makedirs(savedir, exist_ok=True)
            
            file = '{}/visdial_{}_{}.json'.format(os.path.join(opt.datasetdir, str(opt.datasetversion)), str(opt.datasetversion), which_set)
            with open(file) as f:
                j = json.load(f)
        
            self.all_questions = j['data']['questions']
            self.all_answers = j['data']['answers']

            samples = j['data']['dialogs']
            self.N = len(samples)
            
            self.images = []
            self.captions = ['' for x in range(self.N)]
            self.caption_ids = torch.LongTensor(self.N, self.S).fill_(dictionary.UNK_ID)
            self.caption_lengths = torch.LongTensor(self.N).fill_(dictionary.UNK_ID)
            self.questions = torch.LongTensor(self.N, self.D).fill_(dictionary.UNK_ID)
            self.question_ids = torch.LongTensor(self.N, self.D, self.S).fill_(dictionary.UNK_ID)
            self.question_lengths = torch.LongTensor(self.N, self.D).fill_(dictionary.UNK_ID)
            self.answers = torch.LongTensor(self.N, self.D).fill_(dictionary.UNK_ID)
            self.answer_ids = torch.LongTensor(self.N, self.D, self.S).fill_(dictionary.UNK_ID)
            self.answer_lengths = torch.LongTensor(self.N, self.D).fill_(dictionary.UNK_ID)
            if self.withoptions:
                self.gtidxs = torch.LongTensor(self.N, self.D).fill_(-1)
                self.answer_options = torch.LongTensor(self.N, self.D, 100).fill_(dictionary.UNK_ID)
                self.answer_options_ids = torch.LongTensor(self.N, self.D, 100, self.S).fill_(dictionary.UNK_ID)
                self.answer_options_lengths = torch.LongTensor(self.N, self.D, 100).fill_(dictionary.UNK_ID)

            for n, s in enumerate(samples):

                sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(n+1), str(self.N), (n+1)/float(self.N)*100))
                sys.stdout.flush()
                
                if opt.datasetversion < 1.0: 
                    self.images.append('COCO_' + j['split'] + '_' + str(s['image_id']).zfill(12) + '.jpg')
                else:
                    self.images.append('VisualDialog_' + j['split'] + '_' + str(s['image_id']).zfill(12) + '.jpg')
               
                ctokens = word_tokenize(utils.preprocess(s['caption'], nc))
                self.caption_ids[n], self.caption_lengths[n] = utils.pad_to_max_length(ctokens, self.S, dictionary)
                self.captions[n] = s['caption']

                for d, dialog in enumerate(s['dialog']):
                    qtokens = word_tokenize(utils.preprocess(self.all_questions[dialog['question']], nc))
                    self.question_ids[n][d], self.question_lengths[n][d] = utils.pad_to_max_length(qtokens, self.S, dictionary)
                    self.questions[n][d] = dialog['question']
                    if (which_set == 'train' or which_set == 'val') or (which_set == 'test' and d < len(s['dialog'])-1): # if test set, handle last entry
                        atokens = word_tokenize(utils.preprocess(self.all_answers[dialog['answer']], nc))
                        self.answer_ids[n][d], self.answer_lengths[n][d] = utils.pad_to_max_length(atokens, self.S, dictionary)
                        self.answers[n][d] = dialog['answer']

                    if self.withoptions and 'answer_options' in dialog:
                        self.gtidxs[n][d] = dialog['gt_index'] if 'gt_index' in dialog else len(s['dialog'])-1
                        for o, option in enumerate(dialog['answer_options']):
                            otokens = word_tokenize(utils.preprocess(self.all_answers[option], nc))
                            self.answer_options_ids[n][d][o], self.answer_options_lengths[n][d][o] = utils.pad_to_max_length(otokens, self.S, dictionary)
                            self.answer_options[n][d][o] = option

            sys.stdout.write("\n")
           
            # save to file
            torch.save(self.images, os.path.join(savedir, 'images.pt'))
            torch.save(self.captions, os.path.join(savedir, 'captions.pt'))
            torch.save(self.caption_ids, os.path.join(savedir, 'caption_ids.pt'))
            torch.save(self.caption_lengths, os.path.join(savedir, 'caption_lengths.pt'))
            torch.save(self.all_questions, os.path.join(savedir, 'all_questions.pt'))
            torch.save(self.questions, os.path.join(savedir, 'questions.pt'))
            torch.save(self.question_ids, os.path.join(savedir, 'question_ids.pt'))
            torch.save(self.question_lengths, os.path.join(savedir, 'question_lengths.pt'))
            torch.save(self.all_answers, os.path.join(savedir, 'all_answers.pt'))
            torch.save(self.answers, os.path.join(savedir, 'answers.pt'))
            torch.save(self.answer_ids, os.path.join(savedir, 'answer_ids.pt'))
            torch.save(self.answer_lengths, os.path.join(savedir, 'answer_lengths.pt'))
            if self.withoptions:
                torch.save(self.gtidxs, os.path.join(savedir, 'gtidxs.pt'))
                torch.save(self.answer_options, os.path.join(savedir, 'answer_options.pt'))
                torch.save(self.answer_options_ids, os.path.join(savedir, 'answer_options_ids.pt'))
                torch.save(self.answer_options_lengths, os.path.join(savedir, 'answer_options_lengths.pt')) 
                    
    def __getitem__(self, n):

        sample = { 'img' : self.image_loader(os.path.join(self.imgdir, self.images[n])),
                   'img_name' : self.images[n],
                   'caption' : self.captions[n],
                   'caption_ids' : self.caption_ids[n],
                   'caption_length' : self.caption_lengths[n],
                   'questions' : self.questions[n], 
                   'questions_ids' : self.question_ids[n], 
                   'questions_length' : self.question_lengths[n],
                   'answers' : self.answers[n],
                   'answers_ids' : self.answer_ids[n],
                   'answers_length' : self.answer_lengths[n], 
                 }

        if self.withoptions:
            sample['gtidxs'] = self.gtidxs[n]
            sample['answer_options'] = self.answer_options[n]
            sample['answer_options_ids'] = self.answer_options_ids[n]
            sample['answer_options_length'] = self.answer_options_lengths[n]

        return sample

    def __len__(self):
        return self.N

    def image_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return self.transform(img.convert('RGB'))
