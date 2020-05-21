import numpy as np
import matplotlib.pyplot as plt
import time

def drawcar(filename):
    #########读取文件START########
    f = open(filename,"r")
    #########读取文件END########

    ##############将文件按行分割START###########
    plt.ion()
    plt.figure(1)
    x = []
    y = []
    z = []
    for line in f:
        line = line[:-1]
        # print(line)
        ###########处理每一行的字符串START########
        onetime = line.split(" ",1)
        print(onetime)
        duringtime = float(onetime[0])
        carnumber = int(onetime[1])
        x.append(duringtime)
        y.append(carnumber)
        z.append(carnumber+1)
        # print(duringtime)
        # print(duringtime.__class__)
        # print(carnumber)
        # print(carnumber.__class__)
        ###########处理每一行的字符串END########
        ###########绘制折线图START###############
        plt.subplot(221)
        plt.plot(x,y)
        plt.title("car")
        plt.xlabel("duringtime")
        plt.ylabel("carnumber")
        plt.tight_layout()

        plt.subplot(222)
        plt.plot(x, y,'.')
        plt.title("car")
        plt.xlabel("during the time")
        plt.ylabel("number of car")
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(x, z)
        plt.title("Predicted car")
        plt.xlabel("during the time")
        plt.ylabel("Predicted number of car")
        plt.tight_layout()

        plt.pause(0.001)
        ###########绘制折线图END###############
    ##############将文件按行分割END###########

if __name__ == '__main__':
    filename = 'car_.txt'
    drawcar(filename)