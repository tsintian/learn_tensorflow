import pyglet

window = pyglet.window.Window()

label = pyglet.text.label('Hello, world',
font_name='Times New Roman',
font_size = 36,
x=window.width//2, y = window.height //2,
anchor_x = 'center', anchor_y = 'center')
