import numpy as np
from os import makedirs, path
from PIL import Image, ImageDraw
height = 148
width = 75
line_thickness = 2
margin = 5
spacing = 3
circle_radius = 9
left_start = line_thickness+margin
top_start = line_thickness+margin
down_start_offset = int(height/2)-1
line_color = (255,255,255)
back_color = (0, 0, 0)

def get_drawer(im):
    return ImageDraw.Draw(im)

def init_image(name):
    makedirs(path.dirname(name), exist_ok=True)
    im = Image.new('RGB', (width, height), back_color)
    return im

def draw_base(drawer):
    drawer.line(((left_start, down_start_offset), (width-left_start-1, down_start_offset)), fill=line_color, width=line_thickness)

def draw_number(drawer, number, up):
    for (columna,fila), value in np.ndenumerate(number): 
        if value:
            x = left_start+circle_radius+columna*(spacing+2*circle_radius)
            y = (up*down_start_offset)+top_start+circle_radius+fila*(spacing+2*circle_radius)
            drawer.ellipse((x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius), fill=line_color)
            
def draw_piece(piece, name):
    im = init_image(name)
    drawer = get_drawer(im)
    draw_base(drawer)
    draw_number(drawer, piece[0], 0)
    draw_number(drawer, piece[1], 1)
#     im.show()
    im.save(name, "JPEG")

def draw_pieces(pieces):
    i=0
    for piece in pieces:
        name = "images/training-images/"+piece[2]+".jpg"
        draw_piece(piece[:2], name)
        i = i+1

def draw_all_pieces():
    draw_pieces(pieces)

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
for index, value in np.ndenumerate(set_index):
#     print(index, value)
    name = ""
    if value>0:
        name =  ""+str(index[0])+"_"+str(index[1])+"_0"
    else:
        name =  ""+str(index[1])+"_"+str(index[0])+"_1"
    piece = np.array([numbers[index[0]],numbers[index[1]], name])
#     print(piece)
    pieces.append(piece)

pieces = np.array(pieces)
draw_pieces(pieces)

