import math
import os
import re

import torch
import torch.nn.utils.rnn as rnn

from PyTorch import transformer, nlp


class Vocabulary:

    def __init__(self, oov_token):
        # DON'T FORGET TO ADD OOV_TOKEN LATER !!!!!
        self.oov_token = oov_token
        self._unused_value = 0
        self._token_to_value = {}
        self._value_to_token = {}

    def add_token(self, token):
        if token not in self._token_to_value:
            self._token_to_value[token] = self._unused_value
            self._value_to_token[self._unused_value] = token
            self._unused_value += 1

    def value_of_token(self, token):
        if token not in self._token_to_value:
            return self._token_to_value[self.oov_token]
        return self._token_to_value[token]

    def token_of_value(self, value):
        if value not in self._value_to_token:
            return self.oov_token
        return self._value_to_token[value]

    def vocab_size(self):
        return len(self._token_to_value)


class Seq2SeqPreProcessor:

    def __init__(self, src_file, tgt_file,
                 padding_token='<0>', sos_token='<s>', eos_token='</s>', oov_token='<unk>'):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.padding_token = padding_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.oov_token = oov_token
        self.src_sequences = []
        self.tgt_sequences = []
        self.src_vocab = Vocabulary(oov_token)
        self.tgt_vocab = Vocabulary(oov_token)

    @staticmethod
    def _sub_tokenize(super_token):
        tokens = []
        while True:
            # Match ([:alpha:]*)([^:alpha:]?)(.*) in the whole string
            m = re.match(r"^([^\W\d_]*)([\W\d_]?)(.*)$", super_token)
            for group in m.groups()[:-1]:
                if group != '':
                    tokens.append(group)
            super_token = m.groups()[-1]
            if super_token == '':
                break
        return tokens

    @staticmethod
    def tokenize_line(line):
        # to lowercase
        line = line.lower()
        return [sub_token
                for super_token in line.split()
                for sub_token in Seq2SeqPreProcessor._sub_tokenize(super_token)]

    def _read_files(self):
        with open(self.src_file, encoding="utf8") as f:
            for line in f:
                self.src_sequences.append(Seq2SeqPreProcessor.tokenize_line(line))
        with open(self.tgt_file, encoding="utf8") as f:
            for line in f:
                self.tgt_sequences.append(Seq2SeqPreProcessor.tokenize_line(line))

    def _token_counts(self):
        # count the number of occurrence of each token
        src_token_to_freq = {}
        tgt_token_to_freq = {}
        for sequence in self.src_sequences:
            for token in sequence:
                if token not in src_token_to_freq:
                    src_token_to_freq[token] = 1
                else:
                    src_token_to_freq[token] += 1
        for sequence in self.tgt_sequences:
            for token in sequence:
                if token not in tgt_token_to_freq:
                    tgt_token_to_freq[token] = 1
                else:
                    tgt_token_to_freq[token] += 1
        src_token_to_freq = [(token, src_token_to_freq[token])
                             for token in src_token_to_freq]
        tgt_token_to_freq = [(token, tgt_token_to_freq[token])
                             for token in tgt_token_to_freq]
        # sort count of occurrences in descending order
        src_token_to_freq.sort(key=lambda x: x[1], reverse=True)
        tgt_token_to_freq.sort(key=lambda x: x[1], reverse=True)
        return src_token_to_freq, tgt_token_to_freq

    def build_vocab(self, min_token_freq=0):
        self._read_files()
        src_token_to_freq, tgt_token_to_freq = self._token_counts()
        # add PADDING, SOS, EOS and OOV tokens
        for token in (self.padding_token, self.sos_token, self.eos_token, self.oov_token):
            self.src_vocab.add_token(token)
            self.tgt_vocab.add_token(token)
        # add source tokens to src_vocab
        for token, count in src_token_to_freq:
            if count <= min_token_freq:
                break
            self.src_vocab.add_token(token)
        # add target tokens to tgt_vocab
        for token, count in tgt_token_to_freq:
            if count <= min_token_freq:
                break
            self.tgt_vocab.add_token(token)

    def build_vocab_from(self, pre_processor):
        self._read_files()
        self.padding_token = pre_processor.padding_token
        self.sos_token = pre_processor.sos_token
        self.eos_token = pre_processor.eos_token
        self.oov_token = pre_processor.oov_token
        self.src_vocab = pre_processor.src_vocab
        self.tgt_vocab = pre_processor.tgt_vocab

    def _sequence_values(self):
        src_values = [torch.tensor(
            [self.src_vocab.value_of_token(token)
             for token in sequence]).long().cuda()
                      for sequence in self.src_sequences]
        tgt_values = [torch.tensor(
            [self.tgt_vocab.value_of_token(token)
             for token in [self.sos_token] + sequence + [self.eos_token]]).long().cuda()
                      for sequence in self.tgt_sequences]
        return src_values, tgt_values

    def batch_generator(self, batch_size=64, repeat_count=1):
        src_values, tgt_values = self._sequence_values()
        src_values, tgt_values = zip(*sorted(zip(src_values, tgt_values), key=lambda x: len(x[0])))
        size = len(src_values)
        src_padding_value = self.src_vocab.value_of_token(self.padding_token)
        tgt_padding_value = self.tgt_vocab.value_of_token(self.padding_token)
        for _ in range(repeat_count):
            start_index = 0
            while start_index < size:
                end_index = min(size, start_index + batch_size)
                src_values_batch = rnn.pad_sequence(
                    src_values[start_index:end_index], batch_first=True,
                    padding_value=src_padding_value).long().cuda()
                tgt_values_batch = rnn.pad_sequence(
                    tgt_values[start_index:end_index], batch_first=True,
                    padding_value=tgt_padding_value).long().cuda()
                start_index = end_index
                yield nlp.Batch(src_values_batch, tgt_values_batch, src_padding_value, tgt_padding_value)

    def values_to_sequence(self, values_list, is_target_value=True):
        # values_list:   torch.Tensor()
        vocab = self.tgt_vocab if is_target_value else self.src_vocab
        sequences = []
        for values in values_list:
            sequence = []
            for value in values:
                token = vocab.token_of_value(value.item())
                if token == self.eos_token:
                    break
                sequence.append(token)
            sequence = ' '.join(sequence)
            sequences.append(sequence)
        return sequences


BATCH_SIZE = 30
# Prepare vocabulary for the entire data
pp = Seq2SeqPreProcessor(os.path.join('data', 'newstest2014.en'),
                         os.path.join('data', 'newstest2014.de'))
pp.build_vocab()
# Prepare train data batch generator
train_pp = Seq2SeqPreProcessor(os.path.join('data', 'newstest2014_train.en'),
                               os.path.join('data', 'newstest2014_train.de'))
train_pp.build_vocab_from(pp)
del pp
src_padding_value = train_pp.src_vocab.value_of_token(train_pp.padding_token)
tgt_padding_value = train_pp.tgt_vocab.value_of_token(train_pp.padding_token)
src_vocab_size = train_pp.src_vocab.vocab_size()
tgt_vocab_size = train_pp.tgt_vocab.vocab_size()
update_interval = math.ceil(len(train_pp.src_sequences) / BATCH_SIZE)
train_batch_gen = train_pp.batch_generator(batch_size=BATCH_SIZE, repeat_count=100)
# Prepare evaluation data batch generator
eval_pp = Seq2SeqPreProcessor(os.path.join('data', 'newstest2014_eval.en'),
                              os.path.join('data', 'newstest2014_eval.de'))
eval_pp.build_vocab_from(train_pp)
eval_batch_gen = eval_pp.batch_generator(batch_size=BATCH_SIZE)

criterion = nlp.LabelSmoothing(tgt_vocab_size=tgt_vocab_size,
                               tgt_padding_val=tgt_padding_value,
                               smoothing=0.0)
model = transformer.make_model(src_vocab_size, tgt_vocab_size, N=3).cuda()
# Try to load the model parameters from disk
try:
    model.load_state_dict(torch.load(os.path.join('models', 'en_de.model')))
except FileNotFoundError:
    pass
model_opt = nlp.NoamOpt(model.src_embed[0].d_model, 1.0 / update_interval, 1000,
                        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))
# Perform the actual training of the model
model.train()
nlp.run_epoch(train_batch_gen, model,
              transformer.SimpleLossCompute(model.generator, criterion, model_opt,
                                            update_interval=update_interval))
# Save the model parameters to disk
torch.save(model.state_dict(), os.path.join('models', 'en_de.model'))
# Evaluate loss on an independent set of data
model.eval()
print(nlp.run_epoch(eval_batch_gen, model,
                    transformer.SimpleLossCompute(model.generator, criterion, None)))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), start_symbol).type_as(src)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys,
                           transformer.subsequent_mask(ys.size(1)).type_as(src))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(-1).type_as(src)], dim=1)
    return ys


model.eval()
src_batch = next(eval_pp.batch_generator(batch_size=10))
src = src_batch.src
src_mask = src_batch.src_mask
print('\n'.join(train_pp.values_to_sequence(
    greedy_decode(model, src, src_mask, max_len=2*src.size(1),
                  start_symbol=train_pp.src_vocab.value_of_token(train_pp.sos_token)))))
print('-----------------------------------------')
print('\n'.join(train_pp.values_to_sequence(src_batch.trg_y)))
