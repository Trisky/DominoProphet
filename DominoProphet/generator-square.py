import numpy as np
from os import makedirs, path
from PIL import Image, ImageDraw

image_height = 150
image_width = 150
image_size_ratio = 0.3
output_size = (int(round(image_width*image_size_ratio)),int(round(image_height*image_size_ratio)))
height = 148
width = 75
line_thickness = 4
margin = 5
spacing = 3
circle_radius = 9
left_start = line_thickness+margin
top_start = line_thickness+margin
down_start_offset = int(height/2)-1
line_color = (255,255,255)
back_color = (0, 0, 0)

def make_image_sets(train_pref, test_pref, split, step):
    tests = 0
    trains =0
    for theta in range (0,360,step):
        mark = (theta%split)
        if (mark == 0):
            tests = tests+1
            draw_pieces(pieces, test_pref, theta)
        else:
            trains = trains+1
            draw_pieces(pieces, train_pref, theta)
    print(trains, tests)

def get_drawer(im):
    return ImageDraw.Draw(im)

def init_image(name):
    makedirs(path.dirname(name), exist_ok=True)
    im = Image.new('RGB', (width, height), back_color)
    return im

def draw_base(drawer):
    drawer.line(((0, down_start_offset), (width, down_start_offset)), fill=line_color, width=line_thickness)

def draw_number(drawer, number, up):
    for (columna,fila), value in np.ndenumerate(number): 
        if value:
            x = left_start+circle_radius+columna*(spacing+2*circle_radius)
            y = (up*down_start_offset)+top_start+circle_radius+fila*(spacing+2*circle_radius)
            drawer.ellipse((x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius), fill=line_color)
            
def draw_piece(piece, degrees, name):
    img = init_image(name)
    drawer = get_drawer(img)
    draw_base(drawer)
    draw_number(drawer, piece[0], 0)
    draw_number(drawer, piece[1], 1)
#     im.show()
    background = Image.new('RGB', (image_width, image_height), back_color)
    bg_w, bg_h = background.size
    img = img.rotate(degrees, expand=1)
    img_w, img_h = img.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    if (image_size_ratio<1.0):
        background.resize(output_size).save(name, "JPEG")
    else:
        background.save(name, "JPEG")

def draw_pieces(pieces, prefix, degrees):
    i=0
    for piece in pieces:
        name = "images/"+prefix+"/"+piece[2]+"-"+str(degrees)+".jpg"
        draw_piece(piece[:2], degrees, name)
        i = i+1

numbers = np.arange(7, dtype=object)
numbers[0] = np.array([[0,0,0], [0,0,0], [0,0,0]], dtype=np.int8)
numbers[1] = np.array([[0,0,0], [0,1,0], [0,0,0]], dtype=np.int8)
numbers[2] = np.array([[1,0,0], [0,0,0], [0,0,1]], dtype=np.int8)
numbers[3] = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.int8)
numbers[4] = np.array([[1,0,1], [0,0,0], [1,0,1]], dtype=np.int8)
numbers[5] = np.array([[1,0,1], [0,1,0], [1,0,1]], dtype=np.int8)
numbers[6] = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=np.int8)

set_index = np.zeros((7, 7))
indices = np.triu_indices(len(set_index))
set_index[indices] = 1

pieces = []
i=0
for index, value in np.ndenumerate(set_index):
    if value>0:
        name =  ""+str(i).zfill(2)+"_"+str(index[0])+""+str(index[1])
    else:
        continue
    piece = np.array([numbers[index[0]],numbers[index[1]], name])
    pieces.append(piece)
    i = i+1

pieces = np.array(pieces)

make_image_sets("trainingImages", "testImages", 10, 1)
