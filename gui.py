import tkinter
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import vedio

def upload():
    root = tkinter.Tk()
    root.withdraw()
    Filepath = filedialog.askopenfilename()  # 获得选择好的文件
    print('Filepath:', Filepath)
#播放按钮
def play():
    v = vedio.Vedio()
    root = tkinter.Tk()
    root.title("warning")
    root.geometry("300x300+300+300")
    root.resizable(False, False)  # 窗口不可调整大小
    root.update()  # 必须
    scroll = tkinter.Scrollbar()
    scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    text = tkinter.Text(root, width=50, height=15)
    text.pack()
    scroll.config(command=text.yview)
    text.config(yscrollcommand=scroll.set)
    for message in [v.play_vedio()]:
        text.insert(tkinter.INSERT,message)
        text.insert(tkinter.END,'\n')
    tkinter.mainloop()


def set_win_center(root, curWidth='', curHight=''):
    '''
    设置窗口大小，并居中显示
    :param root:主窗体实例
    :param curWidth:窗口宽度，非必填，默认200
    :param curHight:窗口高度，非必填，默认200
    :return:无
    '''
    if not curWidth:
        '''获取窗口宽度，默认200'''
        curWidth = root.winfo_width()
    if not curHight:
        '''获取窗口高度，默认200'''
        curHight = root.winfo_height()
    # print(curWidth, curHight)

    # 获取屏幕宽度和高度
    scn_w, scn_h = root.maxsize()
    # print(scn_w, scn_h)

    # 计算中心坐标
    cen_x = (scn_w - curWidth) / 2
    cen_y = (scn_h - curHight) / 2
    # print(cen_x, cen_y)

    # 设置窗口初始大小和位置
    size_xy = '%dx%d+%d+%d' % (curWidth, curHight, cen_x, cen_y)
    root.geometry(size_xy)

root=tkinter.Tk()
root.title('云城交通')
# root.geometry('400x500')
Filepath=""
root.resizable(False, False)  # 窗口不可调整大小
root.update()  # 必须
set_win_center(root, 400, 600)
# 背景
canvas = tkinter.Canvas(root, width=400,bg="#FFFFFF",height=600, bd=0, highlightthickness=0)

imgpath1 = 'img/1.png'
img1 = Image.open(imgpath1)
photo1 = ImageTk.PhotoImage(img1)

imgpath2 = 'img/2.png'
img2 = Image.open(imgpath2)
photo2 = ImageTk.PhotoImage(img2)

imgpath3 = 'img/3.png'
img3 = Image.open(imgpath3)
photo3 = ImageTk.PhotoImage(img3)

imgpath4 = 'img/4.png'
img4 = Image.open(imgpath4)
photo4 = ImageTk.PhotoImage(img4)

imgpath5 = 'img/5.png'
img5 = Image.open(imgpath5)
photo5 = ImageTk.PhotoImage(img5)

imgpath6 = 'img/6.png'
img6 = Image.open(imgpath6)
photo6 = ImageTk.PhotoImage(img6)
canvas.pack()
b1=tkinter.Button(root,relief='raised', cursor="hand2",image=photo4 ,width=40, command=upload)
b1.pack()
b2=tkinter.Button(root,relief='raised',cursor="hand2",image=photo5, width=40, command=play)
b2.pack()
canvas.create_window(260, 200, width=120, height=45,window=b1)
canvas.create_window(260, 350, width=120, height=45,window=b2)
canvas.create_image(100,10,anchor='nw',image=photo1)
canvas.create_image(50,150,anchor='nw',image=photo2)
canvas.create_image(50,300,anchor='nw',image=photo3)
canvas.create_image(230,420,anchor='nw',image=photo6)

#消息循环
root.mainloop()
