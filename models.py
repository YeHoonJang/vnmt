import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import tqdm

# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

# TODO: 전체적으로 size 다시

# TODO: encoder, decoder GRU 구조 파악하고 새로 짜
class encoderGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(encoderGRUCell, self).__init__()
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
        # input (batch_size, input_size)
        # hx (batch_size, hidden_size)
        # hy (batch_size, hidden_size)
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

# TODO Bidirectional로 수정
class encoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(encoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(encoderGRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(encoderGRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input (batch_size, seqence length, input_size)
        # Output  (batch_size, output_size)

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

                # hidden[layer] = hidden_l

            outs.append(hidden_l)

        out = outs[-1].squeeze()
        out = self.fc(out)

        # TODO out, hidden size 확인
        return out, hidden

class decoderGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(decoderGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        # TODO 여기 input size 확인하기
        self.c2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.he2h = nn.Linear(hidden_size, 3 * hidden_size, bias = bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None, c, he):
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)
        c_t= self.c2h(c)
        he_t = self.he2h(he)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)
        c_reset, c_upd, c_new = c_t.chunk(3, 1)
        he_reset, he_upd, he_new = he_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset + c_reset + he_reset)
        update_gate = torch.sigmoid(x_upd + h_upd + c_upd + he_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new) + c_new + he_new)

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy

class decoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(decoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(decoderGRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(decoderGRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input (batch_size, seqence length, input_size)
        # Output  (batch_size, output_size)

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

                # hidden[layer] = hidden_l

            outs.append(hidden_l)

        out = outs[-1].squeeze()
        out = self.fc(out)

        # TODO out, hidden size 확인
        return out, hidden

class encoder(nn.Module):
    def __init__(self, en_gru, input_size, hidden_size):
        super(encoder, self).__init__()

        self.en_gru = en_gru
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.src_embed = nn.Embedding(input_size, hidden_size)

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

    def forward(self, src, hidden):
        src = self.src_embed(src)
        output, hidden = self.gru(src, hidden)
        c = something # attention 연산
        return output, c #TODO output 뭔지 정확히 확인


class decoder(nn.Module):
    def __init__(self, de_gru, output_size, hidden_size):
        super(decoder, self).__init__()
        self.de_gru = de_gru
        self.input_size = output_size
        self.hidden_size = hidden_size
        self.tgt_embed = nn.Embedding(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, tgt, hidden):
        tgt = self.tgt_embed(tgt)
        output, hidden = self.gru(tgt, hidden)
        return output #TODO 여기도 output 뭔지 정확히 확인


def kl_anneal_function(epoch, k, x0):
    # logistic
    return float(1/(1+np.exp(-k*(epoch-x0))))

def loss_fn(out, target, mu, logv, epoch, k, xo):
    MSE = nn.MSELoss()
    MSE_loss = MSE(out, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1+logv-mu.pow(2)-logv.exp())
    KL_weight = kl_anneal_function(epoch, k, xo)

    return MSE_loss, KL_loss, KL_weight


class VNMT(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, embed_size):
        super(VNMT, self).__init__()

        #TODO: GRU output 어떻게 나오는지 확인 -> hidden state에 대해서 모든 값을 다 출력하는지
        self.encoder = encoder
        self.decoder = decoder
        self.infer = encoder
        self.hidden_size = hidden_size
        self.embed_size = embed_size

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
        h_src, c = nn.AvgPool2d(self.encoder(src))
        h_tgt = nn.AvgPool2d(self.encoder(tgt))

        ## 논문 다시
        hz = self.tanh(self.hz(torch.cat((h_src, h_tgt), 0))) #TODO cat size 다시 보기

        mu = self.hztomean(hz).to(device)
        logv = self.hztologv(hz).to(device)
        z = self.reparameterize(mu, logv).to(device)

        he = self.tanh(self.he(z)).to(device)

        # TODO decoder(encoder 도) -> GRU 구현 후 더 수정
        output = self.decoder(tgt, c, he)

        return output, mu, logz, v

# TODO 여기 분리
encoder = encoderGRU
decoder = decoderGRU

model = VNMT(encoder, decoder, hidden_size=, embed_size=).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            output, mu, logv, z = model(X, y)   # forward

            MSE_loss, KL_loss, KL_weight = loss_fn(output, y, mu, logv, epoch, 0.0025, 2500)   # k=0.0025, x0=2500
            loss = (MSE_loss + KL_loss*KL_weight)
            loss_value = loss.item()
            train_loss += loss_value

            optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total

@torch.no_grad()    #no autograd (backpropagation X)
def evaluate(model, data_loader, device):
    y_list = []
    output_list = []

    model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (X, y) in enumerate(data_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            output, mu, logv, z = model(X)
            MSE_loss, KL_loss, KL_weight = loss_fn(output, y, mu, logv, epoch, 0.0025, 2500)  # k=0.0025, x0=2500
            loss = (MSE_loss + KL_loss * KL_weight)
            loss_value = loss.item()
            valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()
            pbar.update(1)

    return valid_loss/total, y_list, output_list


# Train
start_epoch = 0
epochs = 100  #argparse
print("Start Training..")
for epoch in range(start_epoch, epochs+1):
    print(f"Epoch: {epoch}")
    epoch_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Training Loss: {epoch_loss:.5f}")

    valid_loss, y_list, output_list = evaluate(model, valid_loader, device)
    # rmse = np.sqrt(valid_loss)
    print(f"Validation Loss: {valid_loss:.5f}")
    # print(f'RMSE is {rmse:.5f}')

