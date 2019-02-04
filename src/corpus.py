import os, sys, torch

# user-defined packages
import utils
import visdial as data

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.freq = {}
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.freq[word] = 1
        else:
            self.freq[word] = self.freq[word] + 1
        
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)

    def set_UNK_ID(self, UNK_ID):
        self.UNK_ID = self.word2idx[UNK_ID]

    def loadwordmodel(self, wordembfile, destfile, wordembsize, log, device):
        if not os.path.exists(destfile):
            log.info('loading pre-trained word embeddings from ' + wordembfile + '... (takes several minutes)')
            if os.path.exists(wordembfile) and 'fasttext' in wordembfile:
                from gensim.models import FastText
                wordvectors = FastText.load_fasttext_format(wordembfile)
            elif os.path.exists(wordembfile) and 'glove' in wordembfile:
                wordvectors = utils.load_glove_embeddings(wordembfile)
            else:
                log.error('word embedding model ' + wordembfile + ' cannot be found!')
                sys.exit()
            
            word_embs = [] 
            c=0
            for i, idx in enumerate(self.idx2word):
                try:
                    word_embs.append(torch.from_numpy(wordvectors[idx]).float())
                except KeyError:
                    c+=1
                    word_embs.append(torch.zeros(wordembsize))
           
            log.info('number of words without a pretrained word embedding: ' + str(c) + '/' + str(len(self.idx2word)))
            
            self.word_embs = torch.stack(word_embs)
            self.word_embs[0].fill_(0) # fill embedding for <PAD> with 0s
            torch.save(self.word_embs, destfile)
        else:
            log.info('loading pre-trained word embeddings from ' + wordembfile + '...')
            self.word_embs = torch.load(destfile)
            log.info('loaded pre-trained word vectors successfully!')

        if device>=0:
            self.word_embs = self.word_embs.to('cuda:' + str(device))

    def filterbywordfreq(self, special_tokens, n_freq=5):
        new_word2idx = {}
        new_idx2word = []
        new_freq = {}
        for idx in range(len(self.word2idx)):
            word = self.idx2word[idx]
            if self.freq[word] >= n_freq or word in special_tokens:
                new_idx2word.append(word)
                new_word2idx[word] = len(new_idx2word)-1
                new_freq[word] = self.freq[word]
        self.idx2word = new_idx2word
        self.word2idx = new_word2idx
        self.freq = new_freq

