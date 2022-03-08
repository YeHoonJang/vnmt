import torch
from torch import nn


# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

# TODO: 전체적으로 size 다시


# TODO: encoder, decoder GRU 구조 파악하고 새로 짜
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder = nn.GRU(input_size=, hidden_size=hidden_size, bidirectional=True)

     def forward(self, x):
        out, hidden = self.encoder(x)
        return out, hidden


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.decoder = nn.GRU(input_size=, hidden_size=hidden_size, bidirectional=True)


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