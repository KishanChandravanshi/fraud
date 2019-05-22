import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from matplotlib.pyplot import figure
# Global Constants
img = None
top_left_list = []
bottom_right_list = []
object_list = []

# Constants
image_folder = 'New Data'
save_directory = "Annotations"
obj = 'vortex_center'

def line_select_callback(clk, rls):
	# just to specify that we're using the global variables
    global top_left_list
    global bottom_right_list

    top_left_list.append((int(clk.xdata), int(clk.ydata))) # data when the user has clicked
    bottom_right_list.append((int(rls.xdata), int(rls.ydata))) # data when the user has left the click
    object_list.append(obj) # corresponding object

def on_key_press(event):
    global object_list
    global top_left_list
    global bottom_right_list
    global img
    if event.key == 'q':
    	# if q is presses it will save the coordinates to a file

        x1, y1 = top_left_list[0][0], top_left_list[0][1]
        x2, y2 = bottom_right_list[0][0], bottom_right_list[0][1]
        with open('annotation.txt','a') as file:
            # we're going according to the retinaNet requirements i.e
            # path/to/file,x1,y1,x2,y2,object_name
            # no space in between
            path = str(img.path) + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + str(obj) + "\n"
            file.write(path)
            
        # re-initialising the list for next image
        # Note we can have multiple bounding boxes in a single image
        # which will look like the following
        # path/to/file,x1,y1,x2,y2,object_name,x1',y1',x2',y2',object_name'

        top_left_list = []
        bottom_right_list = []
        object_list = []
        img = None
        plt.close()


def toggle_selector(event):
    toggle_selector.RS.set_active(True)


if __name__ == '__main__':
	# scan through each image and let user annotate them
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1)
        image = cv2.imread(image_file.path)
        # CV2 -> bgr and matplotlib -> rgb, so converting the bgr to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig.set_dpi(200)
        ax.imshow(image)
        # connecting the rectangular selector so that user can draw BB
        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )

        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event', on_key_press)
        plt.show()
