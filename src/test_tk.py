import tkinter as tk
from tkinter.filedialog import *
import sys
import compare
import classifier
import predict
import test_net
import real_time_recognition
from align.align_dataset_mtcnn import main
from align.align_dataset_mtcnn import parse_arguments
from Read_Capture import register_face
from PIL import Image,ImageTk
import real_time_face_recognition
############################################################
root = tk.Tk()
root.title("FaceNet人脸识别Demo")
root.geometry('400x380')
image_file = tk.PhotoImage(file="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\test1.gif")
L12 =tk.Label(root,compound=tk.CENTER,text="     FaceNet\n人脸识别Demo",justify=tk.LEFT,width=400,height=400,image=image_file,fg = "white")
L12.pack()
##########################################################

##########################人脸注册###########################
var5 = tk.StringVar()
def register_ui():
    def open_file1():
        filename1 = askdirectory()
        if filename1!="":
            var5.set(filename1)
    def register():
        register_face(var5.get())
        from tkinter.messagebox import showinfo
        tk.messagebox.showinfo(title="tips", message="注册完毕")
    top = tk.Toplevel(root)
    top.geometry("650x130")
    top.title("人脸注册")



    tk.Label(top, text="请以你的名字新建一个文件夹").grid(row=0, column=0)
    tk.Label(top, text="文件夹路径:").grid(row=1, column=0)
    e1 = tk.Entry(top, textvariable=var5, width=80)
    e1.grid(row=2, column=0)
    tk.Button(top, text="打开文件夹", command=open_file1).grid(row=2, column=1)
    tk.Button(top, text="进行人脸注册", command=register).grid(row=3, column=1)
#########################人脸识别############################
def recognize_svm():
    sys.argv[1:]=[]
    real_time_face_recognition.main(real_time_face_recognition.parse_arguments(sys.argv[1:]))
##########################################################

def recognize_update():
    sys.argv[1:]=[]
    real_time_recognition.main(real_time_recognition.parse_arguments(sys.argv[1:]))

######################训练模型###############################
def train_ui():
    top = tk.Toplevel(root)
    top.geometry("800x120")
    top.title("训练模型")
    var1 = tk.StringVar()
    var2 = tk.StringVar()
    var3 = tk.StringVar()
    var3.set('TRAIN')
    def open_file1():
        filename1 = askdirectory()
        if filename1!="":
            var1.set(filename1)
    def open_file2():
        filename2 = askopenfilename()
        if filename2!="":
            var2.set(filename2)
    def train():
        sys.argv[1:]=[var3.get(),var1.get(),var2.get()]
        classifier.main(classifier.parse_arguments(sys.argv[1:]))
        from tkinter.messagebox import showinfo
        tk.messagebox.showinfo(title="tips", message="训练完毕")

    tk.Label(top, text="模式:").grid(row=0, column=0)
    e3 = tk.Entry(top, state='readonly',textvariable=var3)
    e3.grid(row=0, column=1)
    tk.Label(top, text="输入文件夹路径").grid(row=1, column=0)
    e1 = tk.Entry(top, textvariable=var1, width=80)
    e1.grid(row=1, column=1)
    tk.Button(top, text="打开文件夹", command=open_file1).grid(row=1, column=2)

    tk.Label(top, text="输出文件路径").grid(row=2, column=0)
    e2 = tk.Entry(top, textvariable=var2, width=80)
    e2.grid(row=2, column=1)
    tk.Button(top, text="打开文件", command=open_file2).grid(row=2, column=2)

    tk.Button(top, text="进行模型训练", command=train).grid(row=3, column=2)

    pass

##########################################################
#####################人脸预测################################
def eval_ui():
    var7 = tk.StringVar()
    var2 = tk.StringVar()
    var3 = tk.StringVar()
    var8 = tk.StringVar()
    top = tk.Toplevel(root)
    top.geometry("800x150")
    top.title("人脸预测")

    def open_file2():
        filename2 = askopenfilename()
        if filename2!="":
            var2.set(filename2)
    def open_file3():
        filename3 = askopenfilename()
        if filename3!="":
            var3.set(filename3)
    def eval():
        #sys.argv[1:]=[e2.get()]
        sys.argv[1:] = [e2.get(), e3.get()]
        print(e2.get())
        print(e3.get())
        dic,time1 = predict.main(predict.parse_arguments(sys.argv[1:]))
        var7.set(dic)
        var8.set(time1)
    tk.Label(top, text="图片路径").grid(row=1, column=0)
    e2 = tk.Entry(top, textvariable=var2, width=80)
    e2.grid(row=1, column=1)
    tk.Button(top, text="打开图片", command=open_file2).grid(row=1, column=2)

    tk.Label(top, text="文件路径").grid(row=2, column=0)
    e3 = tk.Entry(top, textvariable=var3, width=80)
    e3.grid(row=2, column=1)
    tk.Button(top, text="打开pkl文件", command=open_file3).grid(row=2, column=2)

    tk.Button(top, text="进行人脸预测", command=eval).grid(row=3, column=2)
    tk.Label(top, text="预测人姓名为:").grid(row=4, column=0)
    l1 = tk.Label(top, textvariable=var7)
    l1.grid(row=4, column=1)

    tk.Label(top, text="花费时间:").grid(row=5, column=0)
    l3 = tk.Label(top, textvariable=var8)
    l3.grid(row=5, column=1)


##########################################################
#####################人脸对齐################################
def align_ui():
    var1 = tk.StringVar()
    var2 = tk.StringVar()
    def open_file1():
        filename1 = askdirectory()
        if filename1!="":
            var1.set(filename1)
    def open_file2():
        filename2 = askdirectory()
        if filename2!="":
            var2.set(filename2)
    def align():
        sys.argv[1:] = [e1.get(), e2.get()]
        main(parse_arguments(sys.argv[1:]))
        from tkinter.messagebox import showinfo
        tk.messagebox.showinfo(title="tips", message="对齐完毕")
    top = tk.Toplevel(root)
    top.geometry("800x120")
    top.title("人脸对齐")

    tk.Label(top, text="输入文件夹路径").grid(row=0, column=0)
    e1 = tk.Entry(top, textvariable=var1, width=80)
    e1.grid(row=0, column=1)
    tk.Button(top, text="打开文件夹", command=open_file1).grid(row=0, column=2)

    tk.Label(top, text="输出文件夹路径").grid(row=1, column=0)
    e2 = tk.Entry(top, textvariable=var2, width=80)
    e2.grid(row=1, column=1)
    tk.Button(top, text="打开文件夹", command=open_file2).grid(row=1, column=2)

    tk.Button(top, text="进行人脸对齐", command=align).grid(row=2, column=2)
#############################################################


#######################人脸比对##############################
def compare_ui():
    var1 = tk.StringVar()
    var2 = tk.StringVar()
    var3 = tk.StringVar()
    var4 = tk.StringVar()
    var5 = tk.StringVar()
    def open_file1():
        filename1 = askopenfilename()
        if filename1!="":
            var1.set(filename1)
    def open_file2():
        filename2 = askopenfilename()
        if filename2!="":
            var2.set(filename2)
    def compare2():
        sys.argv[1:]=[e1.get(),e2.get()]
        dist,time1 = compare.main(compare.parse_arguments(sys.argv[1:]))
        if dist<1:
            var3.set(dist)
            var4.set("判断为同一个人脸")
        else:
            var3.set(dist)
            var4.set("判断为不是同一个人脸")
        var5.set(time1)
    top = tk.Toplevel(root)
    top.geometry("800x150")
    top.title("人脸比对")

    tk.Label(top,text="文件路径1").grid(row=0,column=0)
    e1 =tk.Entry(top,textvariable=var1,width=80)
    e1.grid(row=0,column=1)
    tk.Button(top,text="打开文件1",command=open_file1).grid(row=0,column=2)

    tk.Label(top, text="文件路径2").grid(row=1, column=0)
    e2 = tk.Entry(top,textvariable=var2,width=80)
    e2.grid(row=1, column=1)
    tk.Button(top,text="打开文件2",command=open_file2).grid(row=1,column=2)

    tk.Button(top,text="进行人脸比对",command=compare2).grid(row=2,column=2)

    tk.Label(top, text="两者欧式距离为:").grid(row=3, column=0)
    l1 = tk.Label(top, textvariable=var3)
    l1.grid(row=3, column=1)
    l2 = tk.Label(top, textvariable=var4)
    l2.grid(row=3, column=2)
    tk.Label(top, text="花费时间为:").grid(row=4, column=0)
    l3 = tk.Label(top, textvariable=var5)
    l3.grid(row=4, column=1)
#######################人脸检测#################################
def detect_ui():
    var1 = tk.StringVar()
    var3 = tk.StringVar()
    var4 = tk.StringVar()
    def open_file1():
        filename1 = askopenfilename()
        if filename1 != "":
            var1.set(filename1)
    def detect():
        image = e1.get()
        num,path,time1 = test_net.search_face(image)
        var3.set(num)
        var4.set(time1)
        img_open = Image.open(path)
        img = ImageTk.PhotoImage(img_open)
        l2.config(image=img)
        l2.image = img  # keep a reference

    top = tk.Toplevel(root)
    top.geometry("900x700")
    top.title("人脸检测")

    tk.Label(top, text="文件路径").grid(row=0, column=0)
    e1 = tk.Entry(top, textvariable=var1, width=80)
    e1.grid(row=0, column=1)
    tk.Button(top, text="打开文件", command=open_file1).grid(row=0, column=2)

    tk.Button(top, text="进行人脸检测", command=detect).grid(row=1, column=2)

    tk.Label(top, text="检测到的人脸数为:").grid(row=2, column=0)
    l1 = tk.Label(top, textvariable=var3)
    l1.grid(row=2, column=1)

    l2 = tk.Label(top)
    l2.grid(row=4, column=1)
    tk.Label(top, text="花费时间为:").grid(row=3, column=0)
    l1 = tk.Label(top, textvariable=var4)
    l1.grid(row=3, column=1)
##############################################################


##########################主界面#############################
l1 = tk.Label(root,bg="blue",text="系统功能",width=13,height=2)
l1.place(x=40,y=60)
b1 = tk.Button(root,text="---1.人脸注册---",command=register_ui)
b1.place(x=40, y=120)
b2=tk.Button(root,text="---2.人脸对齐---",command=align_ui)
b2.place(x=40, y=180)
b4=tk.Button(root,text="---3.训练模型---",command=train_ui)
b4.place(x=40, y=240)
b6=tk.Button(root,text="---4.实时识别o--",command=recognize_svm)
b6.place(x=40,y=300)




l2= tk.Label(root,bg="yellow",text="趣味功能",width=13,height=2)
l2.place(x=260,y=60)
b3=tk.Button(root,text="---1.人脸比对---",command=compare_ui)
b3.place(x=260,y=120)
b5=tk.Button(root,text="---2.人脸检测---",command=detect_ui)
b5.place(x=260,y=180)
b9=tk.Button(root,text="---3.人脸预测---",command=eval_ui)
b9.place(x=260,y=240)
b8=tk.Button(root,text="---4.实时识别u--",command=recognize_update)
b8.place(x=260,y=300)
root.mainloop()
##############################################################