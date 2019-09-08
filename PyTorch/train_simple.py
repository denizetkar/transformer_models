from PyTorch import transformer, nlp
import torch
import numpy as np
import torch.nn.utils.rnn as rnn


def data_gen(V, batch, n_batches, max_seq_length):
    # Generate random data for a src-tgt copy task
    # 0 -> padding, 1 -> SOS, 2 -> EOS
    for i in range(n_batches):
        src_seq_list, tgt_seq_list = [], []
        for j in range(batch):
            src_seq_list.append(torch.randint(3, V, (np.random.randint(1, max_seq_length + 1),)))
            tgt_seq_list.append(torch.cat([
                torch.tensor([1]),
                src_seq_list[j],
                torch.tensor([2])]))
        src = rnn.pad_sequence(src_seq_list, batch_first=True, padding_value=0).long().cuda()
        tgt = rnn.pad_sequence(tgt_seq_list, batch_first=True, padding_value=0).long().cuda()
        yield nlp.Batch(src, tgt, 0)


class SimpleLossCompute:
    # A simple loss compute and train function
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.item()


# Train the simple copy task
V = 11
MAX_SEQ_LENGTH = 10
criterion = nlp.LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = transformer.make_model(V, V, N=2).cuda()
model_opt = nlp.NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))

model.train()
nlp.run_epoch(data_gen(V, 200, 300, MAX_SEQ_LENGTH), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
# model.eval()
# print(nlp.run_epoch(data_gen(V, 30, 5), model,
#                     SimpleLossCompute(model.generator, criterion, None)))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys,
                           transformer.subsequent_mask(ys.size(1)).type_as(src))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
    return ys


model.eval()
src = torch.tensor([list(range(3, 11))]).cuda()
src_mask = torch.ones(1, 1, 8).cuda()
print(greedy_decode(model, src, src_mask, max_len=MAX_SEQ_LENGTH, start_symbol=1))
