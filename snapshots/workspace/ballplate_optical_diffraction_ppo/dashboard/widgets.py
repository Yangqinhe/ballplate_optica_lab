"""Tk previews with Pillow-rendered CJK labels on the installed Linux fonts."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk

BACKGROUND = "#101b2a"
PANEL = "#17273a"
FOREGROUND = "#e7eff8"
MUTED = "#aabbd0"
ACCENT = "#63d1c4"
WARNING = "#ffd285"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def text_image(master,text,size=13,color=FOREGROUND,background=PANEL,wrap=0):
    font = ImageFont.truetype(str(FONT_PATH),size) if FONT_PATH.exists() else ImageFont.load_default()
    lines = []
    for paragraph in str(text).split("\n"):
        if wrap:
            current = ""
            for char in paragraph:
                if current and font.getlength(current+char) > wrap:
                    lines.append(current)
                    current = ""
                current += char
            lines.append(current)
        else:
            lines.append(paragraph)
    width = max(1,int(max(font.getlength(line) for line in lines))+5)
    im = Image.new("RGB",(width,max(1,len(lines)*(size+5)+4)),background)
    draw = ImageDraw.Draw(im)
    for index,line in enumerate(lines):
        draw.text((2,index*(size+5)+2),line,font=font,fill=color,anchor="lt")
    return ImageTk.PhotoImage(im,master=master)


class CjkLabel(ttk.Label):
    def __init__(self,master,text="",size=13,color=FOREGROUND,wrap=0,background=PANEL,**kwargs):
        super().__init__(master,**kwargs)
        self.size,self.color,self.wrap,self.background = size,color,wrap,background
        self.previous = None
        self.set(text)

    def set(self,text):
        if text == self.previous:
            return
        self.previous = text
        self.photo = text_image(self,text,self.size,self.color,self.background,self.wrap)
        self.configure(image=self.photo)


class CjkButton(ttk.Button):
    def __init__(self,master,text="",**kwargs):
        super().__init__(master,**kwargs)
        self.set(text)

    def set(self,text):
        self.photo = text_image(self,text,size=13,background="#243b55")
        self.configure(image=self.photo)


class ImagePanel(ttk.Frame):
    def __init__(self,master,title,size=256,caption="",accent=False):
        super().__init__(master,style="Panel.TFrame",padding=7)
        self.size = size
        self.title = CjkLabel(self,title,color=ACCENT if accent else FOREGROUND,size=14)
        self.title.pack(anchor="w",pady=(0,5))
        self.image_label = tk.Label(self,bg="#08101b",bd=0)
        self.image_label.pack()
        self.caption = CjkLabel(self,caption,size=11,color=MUTED,wrap=size)
        self.caption.pack(anchor="w",pady=(5,0))
        self.set_array(np.zeros((size,size),dtype=np.uint8))

    def set_array(self,array,gain=1.,outline=None):
        array = np.asarray(array)
        if gain != 1:
            array = np.clip(array.astype(np.float32)*gain,0,255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB")
        if outline is not None:
            x,y,w,h = outline
            ImageDraw.Draw(image).rectangle((x,y,x+w-1,y+h-1),outline=ACCENT,width=5)
        image.thumbnail((self.size,self.size),Image.Resampling.BOX)
        canvas = Image.new("RGB",(self.size,self.size),"#08101b")
        canvas.paste(image,((self.size-image.width)//2,(self.size-image.height)//2))
        self.photo = ImageTk.PhotoImage(canvas,master=self.image_label)
        self.image_label.configure(image=self.photo)
