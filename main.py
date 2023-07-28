import time
import ntpath
import  cv2
from ultralytics import YOLO
from tkinter import Tk, Canvas, Button, Label, BooleanVar, Checkbutton, filedialog
from PIL import Image as Pil_image
from PIL import ImageTk as Pil_imageTk


class BSC():
    def __init__(self):
        """Class for running GUI.
        """
        self.pTime = 0
        self.cTime = 0
        self.fps = 0

        self.vid_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.image_width = int(self.vid_stream.get(3))
        self.image_height = int(self.vid_stream.get(4))
        self.bg = None
        self.bg_selected = False
        self.show_bg = False
        self.model = YOLO('yolov8n-seg.pt')
        self.save = False
        self.run_gui()

    def run_gui(self):
        """Function for running the GUI.
        """
        self.root = Tk()
        self.root.title("Background-Swap Camera")
        self.root.geometry("425x500")
        self.root.configure(bg="#ffa5a4")

        #widgets        
        self.image_panel = Canvas(self.root, width=400, height=400)  
        self.image_panel.place(x=10, y=10)

        self.bg_button = Button(self.root, text="Select Background",  width=17, command=lambda :self.select_bg())
        self.bg_button.place(x=10, y=430)

        self.enable_button = Button(self.root, text="Enable Background",  width=17, relief="raised", 
                                    command=lambda :self.enable_button_control(self.enable_button))
        self.enable_button.place(x=147, y=430)

        self.save_button = Button(self.root, text="Save Image",  width=17, command=lambda :self.save_image())
        self.save_button.place(x=286, y=430)

        self.image_label = Label(self.root, text="Image Selected: None", width=37, anchor='w')
        self.image_label.place(x=10, y=465)

        self.fps_check = BooleanVar()
        self.fps_button = Checkbutton(self.root, text = "Show FPS", 
                                      variable = self.fps_check,
                                      onvalue = True,
                                      offvalue = False,
                                      bg="#ffa5a4")
        self.fps_button.place(x=310 ,y=465)

        #webcam loop
        self.webcam_func()

        def on_closing():
            """Runs when the GUI is closed
            """
            self.root.update_idletasks()
            self.root.update()
            self.vid_stream.release()
            self.root.quit()
            self.root.destroy()
            return

        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

    def toggle(self, button):
        """Toggles color and relief button.

        Args:
            button (button): button element.

        Returns:
            Bool: True if button is enabled, else False.
        """
        if button.config('relief')[-1] == 'raised':
            button.config(relief="sunken")
            button.config(bg="#DAF7A6")
            return True
        else:
            button.config(relief="raised")
            button.config(bg="#f0f0f0")
            return False

    def enable_button_control(self, button):
        """controls functions for the enable button.

        Args:
            button (button): button element.
        """
        if not self.bg_selected:
            print("No background selected!!")
        else:
            res = self.toggle(button)
            
            if res:
                self.show_bg = True
            else:
                self.show_bg = False

    def select_bg(self):
        """Function to select image for background.
        """

        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

        cur_text = self.image_label.cget("text")
        self.image_label.configure(text="Fetching image. Please wait...")
        filepath = filedialog.askopenfilename(initialdir = "",
                                              title = "Select a File", 
                                              filetypes=[('JPG image','*.jpg')])
        
        if filepath:
            self.bg = cv2.resize(cv2.imread(filepath), (self.image_width, self.image_height))
            self.image_label.configure(text="Image Selected: "+path_leaf(filepath))
            self.bg_selected=True
            self.enable_button.invoke()
        else:
            self.image_label.configure(text=cur_text)

    def make_square(self, im, min_size=400, max_size=400, fill_color=(0, 0, 0, 0)):
        """used to resize an image to be square without distortions.

        Args:
            im (PIL image): image.
            min_size (int, optional): minimum size the output image can be. Defaults to 400.
            max_size (int, optional): maximum size the output image can be. Defaults to 400.
            fill_color (tuple, optional): RGBA value for background fill color. Defaults to (0, 0, 0, 0).

        Returns:
            PIL image: resized image.
        """
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Pil_image.new('RGB', (size, size), fill_color)    
        if(x > y):
            mod = size/x
            x = int(x * mod)
            y = int(y * mod)
        elif(x < y):
            mod = size/y
            x = int(x * mod)
            y = int(y * mod)

        new_im.paste(im.resize((x,y)), (int((size - x) / 2), int((size - y) / 2)))
        new_im = new_im.resize((max_size,max_size))
        return new_im

    def show_image(self, image):
        """draws the image on the canvas.

        Args:
            image (cv2 image): Image to draw.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Pil_image.fromarray(image)
        image = self.make_square(image)
        image = Pil_imageTk.PhotoImage(image = image)
        self.image_panel.create_image(0, 0, image=image, anchor='nw')
        #self.root.update_idletasks()
        self.root.update()

    def save_image(self):
        """Changes the image saving flag to True.
        """
        self.save = True

    def webcam_func(self):
        """where the webcam images are captured, modified, and sent out to displayed.
        Image saving also happnes here.
        """
        success, img = self.vid_stream.read()
        if success:
            if not self.show_bg:
                final = img 
            else:
                res = self.model(img, verbose=False, stream=False)
                classes = list(map(int, res[0].boxes.cls))
                if 0 in classes:
                    person_index = classes.index(0)
                    mask = ((res[0].masks.data[person_index]).numpy() * 255).astype("uint8")
                    inverted_mask = 255 - mask
                    bg_masked = cv2.bitwise_and(self.bg, self.bg, mask=inverted_mask)
                    fg_masked = cv2.bitwise_and(img, img, mask = mask)
                    final = cv2.bitwise_or(fg_masked, bg_masked)
                else:
                    final = self.bg.copy()

            if self.fps_check.get():
                cv2.putText(final, "FPS: "+str(int(self.fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                
            if self.save:
                self.save = False
                file_name = filedialog.asksaveasfile(mode='w', 
                                                     defaultextension=".png", 
                                                     filetypes=[('PNG image','*.png')])
                
                if file_name:
                    cv2.imwrite(file_name.name, final)

            self.show_image(final)
            
        else:
            print("Unable to read video stream")
            

        
        self.cTime = time.time()
        self.fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime
        
        self.root.after(0, self.webcam_func)
    

def main():
    bsc_obj = BSC()


if __name__ == "__main__":
    main()