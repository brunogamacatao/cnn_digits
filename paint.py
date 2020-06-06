import os
import math
import torch
from tkinter import *
from tkinter.colorchooser import askcolor
from main import net, ARQUIVO_REDE

PEN_SIZE = 40.0

class Paint(object):
    def __init__(self):
        self.root = Tk()

        self.reset_button = Button(self.root, text='Novo', command=self.clear)
        self.reset_button.grid(row=0, column=0)

        self.label_digito = Label(self.root, text='Dígito:')
        self.label_digito.grid(row=0, column=1)

        self.campo_digito = Spinbox(self.root, from_=0, to=9)
        self.campo_digito.grid(row=0, column=2)

        self.pasta_destino = StringVar()
        self.pasta_destino.set('digitnet/digitos')

        self.label_pasta = Label(self.root, text='Pasta:')
        self.label_pasta.grid(row=0, column=3)

        self.campo_pasta = Entry(self.root, textvariable=self.pasta_destino)
        self.campo_pasta.grid(row=0, column=4)

        self.save_button = Button(self.root, text='Salvar', command=self.save)
        self.save_button.grid(row=0, column=5)

        self.identifica_button = Button(self.root, text='Identifica', command=self.identifica)
        self.identifica_button.grid(row = 0, column=6)

        self.c = Canvas(self.root, bg='white', width=640, height=640)
        self.c.grid(row=1, columnspan=10)

        self.label_valor = Label(self.root, text='0')
        self.label_valor.grid(row = 1, column=11)
        self.label_valor.config(font=("Courier", 72))

        self.points = set()

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<Button-1>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.identifica)

    def clear(self):
        self.c.delete("all")
        self.points.clear()

    def get_dados(self):
        lines = []
        for y in range(0, 600 + int(PEN_SIZE), int(PEN_SIZE)):
            for x in range(0, 600 + int(PEN_SIZE), int(PEN_SIZE)):
                point = (x,y)
                lines.append(1 if point in self.points else 0)
        return lines

    def save(self):
        lines = self.get_dados()
        pasta_destino = os.path.join(self.pasta_destino.get(), self.campo_digito.get())
        if not os.path.exists(pasta_destino):
            os.makedirs(pasta_destino)
        arquivos = os.listdir(pasta_destino)
        fname = os.path.join(pasta_destino, "imagem_{}.csv".format(len(arquivos)))
        novo_arquivo = open(fname, 'w')
        novo_arquivo.write(','.join(map(lambda x: str(x), lines)))
        novo_arquivo.close()
        self.clear()

    def identifica(self, event = None):
        lines = self.get_dados()
        saida = net(torch.Tensor(lines).float().view(16, 16).unsqueeze(0).unsqueeze(0))
        _, digito = torch.max(saida.data, 1)
        self.label_valor['text'] = str(digito.item())

    def pixelize(self, coord):
        return int(math.floor(coord / PEN_SIZE) * PEN_SIZE)

    def paint(self, event):
        x, y = self.pixelize(event.x), self.pixelize(event.y)
        point = (x,y)
        if point not in self.points:
            self.c.create_rectangle(x, y, x + PEN_SIZE, y + PEN_SIZE, fill='black')
            self.points.add(point)

if __name__ == '__main__':
    net.load_state_dict(torch.load(ARQUIVO_REDE))
    net.eval()
    Paint()
