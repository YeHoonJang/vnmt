import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

# TODO: 전체적으로 size 다시


# TODO: encoder, decoder GRU 구조 파악하고 새로 짜
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hx=None):
        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()
        out = self.fc(out)

        # TODO out, hidden size 확인
        return out, hidden

class encoder(nn.Module):
    def __init__(self, gru):
        super(encoder, self).__init__()

        self.gru = gru

    # TODO 여기 논문 읽고 한번 더 확인!!
    # def get_att_weight(self, dec_output, enc_outputs):  # get attention weight one 'dec_output' with 'enc_outputs'
    #     n_step = len(enc_outputs)
    #     attn_scores = torch.zeros(n_step)  # attn_scores : [n_step]
    #
    #     for i in range(n_step):
    #         attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])
    #
    #     # Normalize scores to weights in range 0 to 1
    #     return F.softmax(attn_scores).view(1, 1, -1)
    #
    # def get_att_score(self, dec_output, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
    #     score = self.attn(enc_output)  # score : [batch_size, n_hidden]
    #     return torch.dot(dec_output.view(-1), score.view(-1))  # inner product make scalar value

    def forward(self, src):
        output, hidden = self.gru(src)





class VNMT(nn.Module):
    def __init__(self, encoder, decoder):
        super(VNMT, self).__init__()

        #TODO: GRU output 어떻게 나오는지 확인 -> hidden state에 대해서 모든 값을 다 출력하는지
        self.encoder = encoder
        self.decoder = decoder
        self.infer = nn.GRU(input_size=, hidden_size=hidden_size, bidirectional=True)

        self.hz = nn.Linear(hidden_size, embed_size)
        self.hztomean = nn.Linear(hidden_size, embed_size)
        self.hztologv = nn.Linear(hidden_size, embed_size)
        self.he = nn.Linear(,)

        self.tanh = torch.tanh()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, src, tgt):
        # scr, tgt to hz
        #TODO: mean pooling -> AvgPool 쓰려면 axis 기준으로 / 아니면 그냥 len 으로 나눠주기
        h_src = nn.AvgPool2d(self.encoder(src))
        h_tgt = nn.AvgPool2d(self.encoder(tgt))

        ## 논문 다시
        hz = self.tanh(self.hz(torch.cat((h_src, h_tgt), 0))) #TODO cat size 다시 보기

        mu = self.hztomean(hz).to(device)
        logv = self.hztologv(hz).to(device)
        z = self.reparameterize(mu, logv).to(device)

        he = self.tanh(self.he(z)).to(device)

        # TODO decoder(encoder 도) -> GRU 구현 후 더 수정
        output = self.decoder(tgt, he)

        return output


model = VNMT(src, tgt)
#TODO: loss 는 따로
optimizer = torch.optim