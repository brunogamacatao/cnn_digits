# coding: utf-8
import os
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# funções para mostrar as imagens
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

ARQUIVO_REDE  = 'conv_digitos.pth'
PASTA_IMAGENS = 'digitos'

def carrega_imagem(fname):
  img = pd.read_csv(fname, header=None).transpose().squeeze().values
  return torch.tensor(img).view(16, 16)

def carrega_imagens(pasta = PASTA_IMAGENS):
  pares = []
  for digito in range(10):
    subpasta = os.path.join(pasta, str(digito))
    for arquivo in os.listdir(subpasta):
      imagem = carrega_imagem(os.path.join(subpasta, arquivo))
      # converte a imagem para o formato de entrada de uma Conv2d
      # o tensor tem que ser do tipo float
      # no formato [batch_size, input_channels, input_height, input_width]
      # unsqueeze(0) cria uma nova dimensão e coloca o conteúdo dentro dela
      imagem = imagem.float().unsqueeze(0).unsqueeze(0)
      pares.append((imagem, digito))
  return pares

# Criação do modelo da rede neural
class CcnModel(nn.Module):
  def __init__(self):
    super(CcnModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.mp = nn.MaxPool2d(2)
    self.fc = nn.Linear(20, 10)

  def forward(self, x):
    in_size = x.size(0)
    x = F.relu(self.mp(self.conv1(x)))
    x = F.relu(self.mp(self.conv2(x)))
    x = x.view(in_size, -1)
    x = self.fc(x)
    #return F.relu(x)
    return F.log_softmax(x)

net = CcnModel() # Criamos uma instância da rede neural

# Critério para cálculo das perdas
criterion = nn.CrossEntropyLoss()

def treina(dados, max_epochs = 100):
  optimizer = optim.Adam(net.parameters(), 0.001)
  for epoch in range(max_epochs):
    total = 0
    for i, d in enumerate(dados):
      entrada, label = iter(d)
      entrada = Variable(entrada, requires_grad=True)
      label = torch.tensor([label])

      optimizer.zero_grad()
      outputs = net(entrada)
      loss = criterion(outputs, label)
      loss.backward()
      optimizer.step()

      _, predicted = torch.max(outputs.data, 1)
      correct = (predicted == label).sum().item()
      total += correct
    
    acuracia = (total / len(dados)) * 100
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
      epoch + 1, max_epochs, i, len(dados), loss.item(), acuracia
    ))
    if acuracia == 100.0:
      break

def img_show(img):
  to_pil = torchvision.transforms.ToPILImage()
  img = to_pil(img)
  plt.imshow(img)
  plt.show()

def carrega_img(caminho):
  entrada = carrega_imagem(caminho)
  entrada = entrada.float().unsqueeze(0)

  input = entrada.unsqueeze(0)
  features = net.mp(net.conv2(F.relu(net.mp(net.conv1(input)))))
  f_img = features.squeeze()
  return f_img

def exibe_imagem():
  images = [carrega_img('digitnet/digitos/0/imagem_0.csv'),
            carrega_img('digitnet/digitos/1/imagem_0.csv'),
            carrega_img('digitnet/digitos/2/imagem_0.csv'),
            carrega_img('digitnet/digitos/3/imagem_0.csv'),
            carrega_img('digitnet/digitos/4/imagem_0.csv'),
            carrega_img('digitnet/digitos/5/imagem_0.csv'),
            carrega_img('digitnet/digitos/6/imagem_0.csv'),
            carrega_img('digitnet/digitos/7/imagem_0.csv'),
            carrega_img('digitnet/digitos/8/imagem_0.csv'),
            carrega_img('digitnet/digitos/9/imagem_0.csv')]

  # for i in range(20):
  #   images.append(f_img[i].unsqueeze(0))
  
  img_show(torchvision.utils.make_grid(images, nrow=10))


if __name__ == '__main__':
  while True:
    print('MENU DE OPÇÕES')
    print('(T)reinar a rede')
    print('(S)alvar a rede')
    print('(E)xibir imagem')
    print('(X) sair')
    opcao = input('Digite sua opção: ').upper()
    if opcao == 'T':
      treina(carrega_imagens())
      print('rede treinada com sucesso')
    elif opcao == 'S':
      torch.save(net.state_dict(), ARQUIVO_REDE)
      print('rede salva com sucesso')
    elif opcao == 'E':
      exibe_imagem()
    elif opcao == 'X':
      break
    else:
      print('Digite uma opção válida')
