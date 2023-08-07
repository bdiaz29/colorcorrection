import tkinter as tk
from tkinter import ttk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
import glob
import os
from tkinter import filedialog

index = 0
pic_list = []
reference_pic_list = []
reference_filepaths = []
current_stats = None


def thumbnail(img, size):
    h, w, c = img.shape
    mx = max(h, w)
    scale = size / mx
    new_w = int(scale * w)
    new_h = int(scale * h)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return new_img


def load_pic():
    global index, pic_list, window, info_lbl
    message = f'{index}/{len(pic_list)}'
    info_lbl.configure(text=message)

    current = pic_list[index]
    img = current.get_pic()
    set_pic(pic_lbl, img)
    window.update()

def apply_last(event):
    global index, pic_list,current_stats
    current = pic_list[index]
    current.external_stats = current_stats
    load_pic()

def apply_to_current(stats):
    global index, pic_list,current_stats
    current = pic_list[index]
    current.external_stats = stats
    current_stats=stats
    load_pic()


def clear_all():
    global index, pic_list
    for pic in pic_list:
        pic.external_stats = None


def apply_to_all(stats):
    global index, pic_list,current_stats
    for pic in pic_list:
        pic.external_stats = stats
    current_stats=stats
    load_pic()


def Reset_Current():
    current = pic_list[index]
    current.external_stats = None
    load_pic()


def Reset_All():
    global index, pic_list
    for pic in pic_list:
        pic.external_stats = None
    load_pic()


def add_color_reference():
    global index, pic_list, sframe, reference_pic_list, reference_filepaths
    current = pic_list[index]
    # already added as reference
    if current.filepath in reference_filepaths:
        return
    rpic = ReferencePIC(sframe.get_frame(), current.filepath)
    reference_pic_list += [rpic]
    reference_filepaths += [current.filepath]


def load_external_reference():
    global index, pic_list, sframe, reference_pic_list, reference_filepaths
    file_path = filedialog.askopenfilename(
        filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.webp')],
        title='Please select an image file'
    )
    if file_path == '':
        return
    if file_path in reference_filepaths:
        return
    rpic = ReferencePIC(sframe.get_frame(), file_path)
    reference_pic_list += [rpic]
    reference_filepaths += [file_path]
    load_pic()


class ReferencePIC:
    def __init__(self, master, filepath):
        img = cv2.imread(filepath)
        mean, stddev = cv2.meanStdDev(img.astype(np.float32) / 255)
        self.stats = (mean, stddev)
        img = thumbnail(img, 320)
        self.frame = Frame(master)
        self.frame.pack()
        # three components the pic label
        # and the two control buttons
        self.pic_label = Label(self.frame)
        self.pic_label.grid(row=0, column=0, padx=5, pady=5)
        set_pic(self.pic_label, img)

        self.btnframe = Frame(self.frame)
        self.btnframe.grid(row=0, column=1, padx=5, pady=5)

        self.apply_to_current_btn = Button(self.btnframe, command=self.apply_to_current, text='Apply To Current')
        self.apply_to_current_btn.grid(row=0, column=1, padx=5, pady=5)

        self.apply_to_all_btn = Button(self.btnframe, command=self.apply_to_all, text='Apply To All')
        self.apply_to_all_btn.grid(row=1, column=1, padx=5, pady=5)

    def apply_to_current(self):
        apply_to_current(self.stats)

    def apply_to_all(self):
        apply_to_all(self.stats)

    def delete_items(self):
        pass


class PIC:
    def __init__(self, filepath):
        self.filepath = filepath
        self.internal_stats = None
        self.external_stats = None

    def get_internal_stats(self):
        if self.internal_stats is None:
            img = cv2.imread(self.filepath).astype(np.float32) / 255
            mean, stddev = cv2.meanStdDev(img)
            self.internal_stats = (mean, stddev)
        return self.internal_stats

    def clear_external_stats(self):
        self.external_stats = None

    def get_full_pic(self):
        if self.external_stats is None:
            img = cv2.imread(self.filepath)
            return img
        else:
            img = cv2.imread(self.filepath).astype(np.float32) / 255
            if self.internal_stats is None:
                mean, stddev = cv2.meanStdDev(img)
                self.internal_stats = (mean, stddev)
            mean1, stddev1 = self.internal_stats
            mean2, stddev2 = self.external_stats
            channels1 = cv2.split(img)
            new_channels = []
            for i in range(len(channels1)):
                channel = channels1[i] - mean1[i][0]
                channel *= stddev2[i][0] / stddev1[i][0]
                channel += mean2[i][0]
                new_channels.append(channel)
            matched_image1 = cv2.merge(new_channels)
            matched_image1 = np.clip(matched_image1, 0, 1)
            matched_image1 = (matched_image1 * 255).astype(np.uint8)
            return matched_image1

    def get_pic(self, size=768):

        if self.external_stats is None:
            img = cv2.imread(self.filepath)
            img_resize = thumbnail(img, 768)
            return img_resize
        else:
            img = cv2.imread(self.filepath).astype(np.float32) / 255
            if self.internal_stats is None:
                mean, stddev = cv2.meanStdDev(img)
                self.internal_stats = (mean, stddev)
            mean1, stddev1 = self.internal_stats
            mean2, stddev2 = self.external_stats
            channels1 = cv2.split(img)
            new_channels = []
            for i in range(len(channels1)):
                channel = channels1[i] - mean1[i][0]
                channel *= stddev2[i][0] / stddev1[i][0]
                channel += mean2[i][0]
                new_channels.append(channel)
            matched_image1 = cv2.merge(new_channels)
            matched_image1 = np.clip(matched_image1, 0, 1)
            matched_image1 = (matched_image1 * 255).astype(np.uint8)
            img_resize = thumbnail(matched_image1, 768)
            return img_resize


class ScrollableFrame(tk.Frame):
    def __init__(self, master=None, width=100, height=100, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=1, width=width, height=height)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def get_frame(self):
        return self.scroll_frame


def set_pic(label, img_bgr):
    img_rgb = img_bgr[:, :, ::-1]
    img = Image.fromarray(img_rgb)
    photo = ImageTk.PhotoImage(img)
    label.config(image=photo)
    label.image = photo


def clear_pic_list():
    global index, pic_list
    index = 0
    pic_list = []


def load_imgs():
    global pic_list, index
    print("save")
    directory = filedialog.askdirectory()
    if directory == "":
        return
    clear_pic_list()
    filelist = []
    filelist.extend(glob.glob(directory + '/*.jpg'))
    filelist.extend(glob.glob(directory + '/*.png'))
    filelist.extend(glob.glob(directory + '/*.jpeg'))
    filelist.extend(glob.glob(directory + '/*.webp'))
    if len(filelist) == 0:
        return
    for file in filelist:
        pic = PIC(file)
        pic_list += [pic]
    load_pic()


def prev(event):
    global index
    tmp = index - 1
    if tmp >= 0:
        index = tmp
        load_pic()


def next(event):
    global index, pic_list
    tmp = index + 1
    if tmp < len(pic_list):
        index = tmp
        load_pic()


def save_images():
    global pic_list
    print("save")
    directory = filedialog.askdirectory()
    if directory == "":
        return

    for pic in pic_list:
        img = pic.get_full_pic()
        original_file = pic.filepath
        basename = os.path.basename(original_file)
        new_path = f'{directory}/{basename}'
        cv2.imwrite(new_path, img)


# Create a new Tk window
blank = np.ones((768, 768, 3), dtype=np.uint8) * 255
window = Tk()

window.bind('a', prev)
window.bind('d', next)
window.bind('<space>', apply_last)
# window.geometry('1600x1600')
window.state('zoomed')
# SCRL frame
SCRL_frame = Frame(window)
SCRL_frame.grid(row=0, column=0, padx=4, pady=4)
# Example label in SCRL frame
# PIC frame
PIC_frame = Frame(window)
PIC_frame.grid(row=0, column=1, padx=4, pady=4)
CTRL_frame = Frame(window)
CTRL_frame.grid(row=0, column=2, padx=4, pady=4)

ldtn = Button(CTRL_frame, command=load_imgs, text="Load Images")
ldtn.grid(row=0, column=0, padx=5, pady=5)

addbtn = Button(CTRL_frame, command=add_color_reference, text="Add Current as Reference")
addbtn.grid(row=1, column=0, padx=5, pady=5)

addbtn2 = Button(CTRL_frame, command=load_external_reference, text="Add External as Reference")
addbtn2.grid(row=2, column=0, padx=5, pady=5)

clrbtn = Button(CTRL_frame, command=Reset_Current, text="Revert Current")
clrbtn.grid(row=3, column=0, padx=5, pady=5)

clrbtn = Button(CTRL_frame, command=Reset_All, text="Revert All")
clrbtn.grid(row=4, column=0, padx=5, pady=5)

svbtn = Button(CTRL_frame, command=save_images, text="Save Images")
svbtn.grid(row=5, column=0, padx=5, pady=5)

pic_lbl0 = Label(PIC_frame, text="Current Pic")
pic_lbl0.grid(row=0, column=0)
pic_lbl = Label(PIC_frame)
pic_lbl.grid(row=1, column=0)

info_lbl = Label(PIC_frame)
info_lbl.grid(row=2, column=0)

set_pic(pic_lbl, blank)
scrl_lbl = Label(SCRL_frame, text="Reference Pics")
scrl_lbl.grid(row=0, column=0)  # adjust the row and column as needed
sframe = ScrollableFrame(SCRL_frame, width=512, height=768, )
sframe.grid(row=1, column=0, padx=5, pady=5)

# Loop through the Tk window instance
window.mainloop()
