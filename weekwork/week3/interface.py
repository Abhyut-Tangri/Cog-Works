# create a interface

import tkinter as tk 

from PIL import Image, ImageTk



root=tk.Tk()
root.title("Semantic Image Searcher")

entry= tk.Entry(root)

entry.pack()

search_button= tk.Button(root, text="Search Image")
search_button.pack()

image_label = tk.Label(root, text="Image will be displayed here")
image_label.pack()

root.mainloop()