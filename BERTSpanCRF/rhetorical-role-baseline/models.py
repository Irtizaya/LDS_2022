from allennlp.common.util import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import masked_mean, masked_softmax
import copy

from transformers import BertModel

from allennlp.modules import ConditionalRandomField

import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time

from allennlp.modules.span_extractors import EndpointSpanExtractor

class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''

        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label

            #log_denominator = self.crf._input_likelihood(logits, mask)
            #log_numerator = self.crf._joint_likelihood(logits, predicted_label, mask)
            #log_likelihood = log_numerator - log_denominator
            #outputs["log_likelihood"] = log_likelihood

        return outputs


class SpanCRF():
  def __init__(self, label_to_ind, max_path):
        super(SpanCRF, self).__init__()
            
        self.tag_to_ix = label_to_ind
        self.tagset_size = len(self.tag_to_ix)
        self.max_path = max_path
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
  def _forward_alg(self, logits, len_list, is_volatile=False):
        """
        Computes the (batch_size,) denominator term (FloatTensor list) for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, max_path, n_labels = logits.size()
        
        alpha = logits.data.new(batch_size, seq_len+1, self.tagset_size).fill_(-10000)
        alpha[:, 0, self.tag_to_ix['START']] = 0
        alpha = Variable(alpha, volatile=is_volatile)
        
        # Transpose batch size and time dimensions:
        logits_t = logits.permute(1,0,2,3)
        c_lens = len_list.clone()
        
        alpha_out_sum = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(0))
        mat = Variable(logits.data.new(batch_size,self.tagset_size,self.tagset_size).fill_(0))
        
        for j, logit in enumerate(logits_t):
            for i in range(0,max_path):
                if i<=j:
                    alpha_exp = alpha[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                    logit_exp = logit[:, i].unsqueeze(-1).expand(batch_size, self.tagset_size, self.tagset_size)
                    trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
                    mat = alpha_exp + logit_exp + trans_exp
                    alpha_out_sum[:,i,:] =  self.log_sum_exp(mat , 2, keepdim=True)
                    
            alpha_nxt = self.log_sum_exp(alpha_out_sum , dim=1, keepdim=True).squeeze(1)
            
            mask = Variable((c_lens > 0).float().unsqueeze(-1).expand(batch_size,self.tagset_size))
            alpha_nxt = mask * alpha_nxt + (1 - mask) *alpha[:, j, :].clone() 
            
            c_lens = c_lens - 1      

            alpha[:,j+1, :] = alpha_nxt

        alpha[:,-1,:] = alpha[:,-1,:] + self.transitions[self.tag_to_ix['STOP']].unsqueeze(0).expand_as(alpha[:,-1,:])
        norm = self.log_sum_exp(alpha[:,-1,:], 1).squeeze(-1)

        return norm
  

  def viterbi_decode(self, logits, lens):
      """
      Use viterbi algorithm to compute the most probable path of segments
      
      Arguments:
          logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
          lens: [batch_size] LongTensor
      """
      batch_size, seq_len, max_path, n_labels = logits.size()
      
      # Transpose to batch size and time dimensions
      logits_t = logits.permute(1,0,2,3)
      
      vit = Variable(logits.data.new(batch_size,seq_len+1, self.tagset_size).fill_(-10000),
                                      volatile = not self.training)
      
      vit_tag_max = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(-10000),
                                  volatile = not self.training) 
      
      vit_tag_argmax = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(-100),
                                  volatile = not self.training) 
      vit[:,0, self.tag_to_ix['START']] = 0
      c_lens = Variable(lens.clone(), volatile= not self.training)
      
      pointers = Variable(logits.data.new(batch_size, seq_len, self.tagset_size, 2 ).fill_(-100))
      for j, logit in enumerate(logits_t):
          for i in range(0,max_path):
              if i<=j:
                  vit_exp = vit[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                  trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
                  vit_trn_sum = vit_exp + trn_exp
                  vt_max, vt_argmax = vit_trn_sum.max(2)
                  vit_nxt = vt_max + logit[:, i]
                  vit_tag_max[:,i,:] = vit_nxt
                  vit_tag_argmax[:,i,:] = vt_argmax
          
          seg_vt_max, seg_vt_argmax = vit_tag_max.max(1)
          
          mask = (c_lens > 0).float().unsqueeze(-1).expand_as(seg_vt_max)
          vit[:, j+1, :] = mask*seg_vt_max + (1-mask)*vit[:, j, :].clone()
          
          mask = (c_lens == 1).float().unsqueeze(-1).expand_as(  vit[:, j+1, :])
          vit[:, j+1, :] = vit[:, j+1, :] +  mask * self.transitions[ self.tag_to_ix['STOP'] ].unsqueeze(0).expand_as( vit[:, j+1, :] )
          
          idx_exp = seg_vt_argmax.unsqueeze(1)
          pointers[:,j,:,0] =  torch.gather(vit_tag_argmax, 1,idx_exp ).squeeze(1)
          pointers[:,j,:,1] = seg_vt_argmax 
          
          c_lens = c_lens - 1  
      
      #Get the argmax from the last viterbi scores and follow the reverse pointers for the best path 
      end_max , end_max_idx = vit[:,-1,:].max(1)
      end_max_idx = end_max_idx.data.cpu().numpy()
      
      pointers = pointers.data.long().cpu().numpy()
      pointers_rev = np.flip(pointers,1)
      paths = []
      segments = []
      
      for b in range(batch_size):
          #Different lengths each sentence, so get the starting index on the reverse list
          start_index = seq_len-lens[b] 
          path = [end_max_idx[b]]
          segment = [lens[b]]
          
          if (start_index >= seq_len -1):
              paths.append(path)
              continue
          
          max_tuple = pointers_rev[b,start_index,end_max_idx[b]]
          start_index += 1
          prev_tag = end_max_idx[b]
          next_tag = max_tuple[0]
          next_jump = max_tuple[1]
          
          for j, argmax in enumerate(pointers_rev[b,start_index:,:]):
              #Append same tag as many times as indicated by the best segment length we stored
              if next_jump > 0:
                  next_jump -= 1
                  path.insert(0, prev_tag)
                  continue
              #Switch to next tag when we hit zero
              else:
                  segment.insert(0, lens[b]- j-1)
                  path.insert(0, next_tag)
              
              #Get the next tag, and the number of times we have to append the previous one
              prev_tag = next_tag
              max_tuple = argmax[next_tag]
              next_tag = max_tuple[0]
              next_jump = max_tuple[1]
              
          segments.append(segment)     
          paths.append(path)
          
      return paths, segments


  def _bilstm_score(self, logits, labels, seg_inds, lens):
      
      """
      Computes the (batch_size,) numerator (FloatTensor list) for the log-likelihood, which is the
      
      Arguments:
          logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
          labels: [batch_size, seq_len] LongTensor
          seg_inds: [batch_size, seq_len] LongTensor
          lens: [batch_size] LongTensor
      """
      lens = Variable( lens, volatile = not self.training)
      
      batch_size, max_len, _, _ = logits.size()
      
      # Transpose to batch size and time dimensions
      labels = labels.transpose(1,0)
      
      seg_inds = seg_inds.transpose(1,0).data.cpu().numpy()
      labels_exp = labels.unsqueeze(-1)

      #Construct the mask the will sellect the corrects segments from all possible segments for each timstep
      mask_seg = np.zeros(( batch_size, max_len, self.max_path))
      
      mask_step =  np.zeros(( batch_size), dtype=np.int32)
      counter = np.zeros((batch_size), dtype=np.int32)
      
      #For each timstep accross all sentences
      for i in range(0,max_len):
          #0 or 1 depending if we are on the end of a segment
          mask_step =  seg_inds[:, i] 
          mask_seg[np.arange(batch_size), i, counter] = mask_step 
          counter = counter + 1
          counter = (1- mask_step)*counter*(counter < self.max_path)
          
      mask_seg = torch.from_numpy(mask_seg).float()
      if next(self.parameters()).is_cuda == True:
          mask_seg = mask_seg.cuda()
          
      mask_seg = mask_seg.unsqueeze(-1).expand_as(logits)
      mask_seg = Variable(mask_seg,  volatile = not self.training) 
      
      logit_mask = logits*mask_seg
      sum_cols = torch.sum(logit_mask, dim=2).squeeze(2)
      
      all_scores = torch.gather(sum_cols, 2, labels_exp).squeeze(-1)
      
      mask_time = self.sequence_mask(lens).float()
      all_scores = all_scores*mask_time
      
      sum_seg_scores = torch.sum(all_scores, dim=1).squeeze(-1)

      return  sum_seg_scores

  def score(self, logits, y, seg_inds, lens):


      bilstm_score = self._bilstm_score(logits, y, seg_inds, lens)
      transition_score = self.transition_score(y, lens, seg_inds )
      
      score = transition_score + bilstm_score

      return score
  
  def transition_score(self, labels, lens, mask_seg_idx):
      """
      Computes the (batch_size,) scores (FloatTensor list) that will be added to the emission scores
      
      Arguments:
          logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
          labels: [batch_size, seq_len] LongTensor
          seg_inds: [batch_size, seq_len] LongTensor
          lens: [batch_size] LongTensor
      """
      lens = Variable( lens, volatile = not self.training)
      labels = labels.transpose(1,0)
      mask_seg_idx = mask_seg_idx.transpose(1,0)
      batch_size, seq_len = labels.size()
      # pad labels with <start> and <stop> indices
      labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
      labels_ext[:, 0] = self.tag_to_ix['START']
      labels_ext[:, 1:-1] = labels
      mask = self.sequence_mask(lens + 1, max_len=seq_len + 2).long()
      pad_stop = Variable(labels.data.new(1).fill_(self.tag_to_ix['STOP']))
      
      pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
      labels_ext = (1 + (-1)*mask) * pad_stop + mask * labels_ext
      trn = self.transitions
      
      trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
      lbl_r = labels_ext[:, 1:]
      lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
      trn_row = torch.gather(trn_exp, 1, lbl_rexp)
      
      lbl_lexp = labels_ext[:, :-1].unsqueeze(-1)
      trn_scr = torch.gather(trn_row, 2, lbl_lexp)
      trn_scr = trn_scr.squeeze(-1)
      
      # Mask sentences in time dim
      mask = self.sequence_mask(lens + 1).float()
      trn_scr = trn_scr * mask
      
      trn_scr[:, 1:] = trn_scr[:, 1:].clone()*mask_seg_idx.float() 
      
      score = trn_scr.sum(1).squeeze(-1)
      
      return score

  def loglik(self, logits, y, lens):
      norm_score = self._forward_alg(logits, lens)
      sequence_score = self.score(logits, y, lens, logits=logits)
      loglik = sequence_score - norm_score

      return loglik   


  def log_sum_exp(vec, dim=0, keepdim=True):
      max_val, idx = torch.max(vec, dim, keepdim=True)
      max_exp = max_val.expand_as(vec)
      
      return max_val + torch.log(torch.sum(torch.exp(vec - max_exp), dim, keepdim=keepdim))

      
  def sequence_mask(lens, max_len=None):
      batch_size = lens.size(0)
      if max_len is None:
          
          max_len = lens.max().data[0]
              
      ranges = torch.arange(0, max_len).long()
      ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
      ranges = Variable(ranges)
      if lens.data.is_cuda:
          ranges = ranges.cuda()

      lens_exp = lens.unsqueeze(1).expand_as(ranges)
      mask = ranges < lens_exp

      return mask

class CRFPerTaskOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks):
        super(CRFPerTaskOutputLayer, self).__init__()

        self.per_task_output = torch.nn.ModuleDict()
        for task in tasks:
            self.per_task_output[task.task_name] = CRFOutputLayer(in_dim=in_dim, num_labels=len(task.labels))


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.per_task_output[task](x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, task in enumerate(self.per_task_output.keys()):
            if index % 2 == 0:
                self.task_to_device[task] = device1
                self.per_task_output[task].to(device1)
            else:
                self.task_to_device[task] = device2
                self.per_task_output[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]



class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        #shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s



class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        # state_dict_1 = self.bert.state_dict()
        # state_dict_2 = torch.load('/home/astha_agarwal/model/pytorch_model.bin')
        # for name2 in state_dict_2.keys():
        #    for name1 in state_dict_1.keys():
        #        temp_name = copy.deepcopy(name2)
        #       if temp_name.replace("bert.", '') == name1:
        #            state_dict_1[name1] = state_dict_2[name2]

        #self.bert.load_state_dict(state_dict_1,strict=False)

        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]

        if not self.bert_trainable and batch["task"] in self.cacheable_tasks:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, num_labels):
        super(BertHSLN, self).__init__()

        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.sentence_lstm_hidden_size = config["sentence_lstm_hs"]
        input_dim = self.attention_pooling.output_dim
        print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=self.sentence_lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

        #self.reinit_output_layer(tasks, config)

        self.endpoint_span_extractor = EndpointSpanExtractor(self.sentence_lstm_hidden_size * 2, 
                                                             combination = "x,y,x*y,x-y", 
                                                             num_width_embeddings = config["max_path"], 
                                                             span_width_embedding_dim = config["span_width_embedding_dim"], 
                                                             bucket_widths = True)
        
        self.input_dim  = self.sentence_lstm_hidden_size * 2
        self.max_path = config["max_path"]
        self.num_labels =  num_labels
        
        
        self._span_crf = config["_span_crf"]
        self._crf = config["_crf"]

        if self._crf:
          self.crf_fc  = nn.Linear(self.input_dim, num_labels)
          self.crf = CRFOutputLayer(in_dim  = self.input_dim, num_labels = num_labels)

        if self._span_crf:
          self.span_input_dim = self.sentence_lstm_hidden_size * 2 * 4 * config["span_width_embedding_dim"]
          self.crf_spanfc = nn.Linear(self.span_input_dim, self.num_labels)
          self.spancrf = SpanCRF(config["label_to_ind"], self.max_path)



    # def init_sentence_enriching(self, config, tasks):
    #     self.sentence_lstm_hidden_size = config["sentence_lstm_hs"]
    #     input_dim = self.attention_pooling.output_dim
    #     print(f"Attention pooling dim: {input_dim}")
    #     self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
    #                               hidden_size=self.sentence_lstm_hidden_size,
    #                               num_layers=1, batch_first=True, bidirectional=True))

    # def reinit_output_layer(self, tasks, config):
    #     if config.get("without_context_enriching_transfer"):
    #         self.init_sentence_enriching(config, tasks)
    #     input_dim = self.lstm_hidden_size * 2

    #     if self.generic_output_layer:
    #         self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
    #     else:
    #         self.crf = CRFPerTaskOutputLayer(input_dim, tasks)

    def forward(self, batch, labels=None, output_all_tasks=False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)


        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)


        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)
        print("Sentence embedding encoded dim ", sentence_embeddings_encoded.shape)

        sentence_len = torch.sum(sentence_mask, dim  = 1)
        output  = {}

        if self._span_crf:
          print("SPAN id, SENTENCE ", batch["span_indices"].shape, sentence_mask)
          span_embeddings =  self.endpoint_span_extractor(sentence_embeddings_encoded, batch["span_indices"], sentence_mask)
          segment_rep  = self.crf_spanfc(span_embeddings)
          _, max_span_len, _ = segment_rep.shape

          segment_span_feat  = torch.zeros(documents, sentences, self.max_path, self.num_labels)

          batch_size, max_span_len, _ = batch["span_indices"].shape
          _, max_seq_len, max_path_len, _ = segment_span_feat.shape

          for i in range(batch_size):
            for j in range(max_span_len):
              start_idx  = batch["span_indices"][i][j][0]
              len_idx  = batch["span_indices"][i][j][1] - start_idx
              segment_span_feat[i, start_idx, len_idx, :] = segment_rep[i][j]

          segment_mask = batch["segment_mask"]

          span_forward_var_batch = self.spancrf._forward_alg(segment_span_feat, sentence_len)
          span_gold_score_batch = self.spancrf.score(segment_span_feat, labels.transpose(0,1), segment_mask.transpose(0,1), sentence_len)
          output['span_crf'] = {"forwrd_var_batch": span_forward_var_batch, "gold_score_batch": span_gold_score_batch}

        if self._crf:
          segment_feat  = sentence_embeddings_encoded.unsqueeze(2)
          segment_feat = self.crf_fc(segment_feat)
          segment_feat  = segment_feat.view(documents, sentences, 1, self.num_labels)

          forward_var_batch = self.crf._forward_alg(segment_feat, sentence_len)
          gold_score_batch = self.crf.score(segment_feat, labels.transpose(0,1), sentence_mask.transpose(0,1), sentence_len)
          output['crf'] = {"forwrd_var_batch": forward_var_batch, "gold_score_batch": gold_score_batch}

        if eval:
          if self._crf:
            crf_tag_seqs, crf_segments = self.crf.viterbi_decode(segment_feat, sentence_len)
            output['crf'] = {"tag_seqs": crf_tag_seqs, "segments": crf_segments}
          if self._span_crf:
            span_crf_tag_seqs, span_crf_segments = self.spancrf.viterbi_decode(segment_span_feat, sentence_len)
            output['span_crf'] = {"tag_seqs": span_crf_tag_seqs, "segments": span_crf_segments}


        # if self.generic_output_layer:
        #     output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        # else:
        #     output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)


        return output

class BertHSLNMultiSeparateLayers(torch.nn.Module):
    '''
    Model Multi-Task Learning, where only certail layers are shared.
    This class is necessary to separate the model on two GPUs.
    '''
    def __init__(self, config, tasks):
        super(BertHSLNMultiSeparateLayers, self).__init__()


        self.bert = BertTokenEmbedder(config)


        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = PerTaskGroupWrapper(
                                        task_groups=config["attention_groups"],
                                        create_module_func=lambda g:
                                            AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])
                                )

        attention_pooling_output_dim = next(iter(self.attention_pooling.per_task_mod.values())).output_dim
        self.sentence_lstm = PerTaskGroupWrapper(
                                    task_groups=config["context_enriching_groups"],
                                    create_module_func=lambda g:
                                    PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=attention_pooling_output_dim,
                                        hidden_size=self.lstm_hidden_size,
                                        num_layers=1, batch_first=True, bidirectional=True))
                                    )

        self.crf = CRFPerTaskGroupOutputLayer(self.lstm_hidden_size * 2, tasks, config["output_groups"])



    def to_device(self, device1, device2):
        self.bert.to(device1)
        self.word_lstm.to(device1)
        self.attention_pooling.to_device(device1, device2)
        self.sentence_lstm.to_device(device1, device2)
        self.crf.to_device(device1, device2)
        self.device1 = device1
        self.device2 = device2

    def forward(self, batch, labels=None, output_all_tasks=False):
        task_name = batch["task"]
        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        device = self.attention_pooling.get_device(task_name)
        sentence_embeddings = self.attention_pooling(task_name, bert_embeddings_encoded.to(device), tokens_mask.to(device))
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        sentence_mask = batch["sentence_mask"]
        # shape: (documents, sentence, 2*lstm_hidden_size)
        device = self.sentence_lstm.get_device(task_name)
        sentence_embeddings_encoded = self.sentence_lstm(task_name, sentence_embeddings.to(device), sentence_mask.to(device))
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        device = self.crf.get_device(task_name)
        if labels is not None:
            labels = labels.to(device)

        output = self.crf(task_name, sentence_embeddings_encoded.to(device), sentence_mask.to(device), labels, output_all_tasks)

        return output

class CRFPerTaskGroupOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks, task_groups):
        super(CRFPerTaskGroupOutputLayer, self).__init__()

        def get_task(name):
            for t in tasks:
                if t.task_name == name:
                    return t

        self.crf = PerTaskGroupWrapper(
                                        task_groups=task_groups,
                                        create_module_func=lambda g:
                                            # we assume same labels per group
                                            CRFOutputLayer(in_dim=in_dim, num_labels=len(get_task(g[0]).labels))
        )
        self.all_tasks = [t for t in [g for g in task_groups]]


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.crf(task, x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for task in self.self.all_tasks:
                task_result = self.crf(task, x, mask, labels)
                task_result["task"] = task
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.crf.to_device(device1, device2)

    def get_device(self, task):
        return self.crf.get_device(task)


class PerTaskGroupWrapper(torch.nn.Module):
    def __init__(self, task_groups, create_module_func):
        super(PerTaskGroupWrapper, self).__init__()

        self.per_task_mod = torch.nn.ModuleDict()
        for g in task_groups:
            mod = create_module_func(g)
            for t in g:
                self.per_task_mod[t] = mod

        self.task_groups = task_groups

    def forward(self, task_name, *args):
        mod = self.per_task_mod[task_name]
        return mod(*args)

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, tasks in enumerate(self.task_groups):
            for task in tasks:
                if index % 2 == 0:
                    self.task_to_device[task] = device1
                    self.per_task_mod[task].to(device1)
                else:
                    self.task_to_device[task] = device2
                    self.per_task_mod[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]


