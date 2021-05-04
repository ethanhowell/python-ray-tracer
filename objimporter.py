import random

colors = [
    '1 0 0',
    '1 .647 0',
    '1 1 0',
    '0 .5 0',
    '0 0 1',
    '.294 0 .51',
    '.933 .51 .933'
]

with open('teapot.obj') as file:
    vertices = [None]
    triangles = []
    for line in file.readlines():
        line = line.split()
        if line[0] == 'v':
            vertices.append(line[1:])
        elif line[0] == 'f':
            v1 = vertices[int(line[1])]
            v2 = vertices[int(line[2])]
            v3 = vertices[int(line[3])]

            color = colors[len(triangles) % len(colors)]

            triangles.append(f'Triangle {" ".join(v1)} {" ".join(v2)} {" ".join(v3)} Kd 0.8 Ks 0.1 Ka 0.1 Od {color} Os {color} Kgls 4')
    for t in triangles:
        print(t)