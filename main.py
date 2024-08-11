import argparse
import numpy as np
import FluidGrid
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.autolayout'] = True
fig, ax = plt.subplots(1,1)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
tx = ax.set_title('start')

img = None
q = None
dynamic_color = True
vel_show = True
fg = None

def render(step):
    if dynamic_color:
        global img, fg
        fg.mass = FluidGrid.goodiv(fg.mass)
        img = ax.imshow(fg.mass,
                        cmap='coolwarm',
                        origin='lower',
                        norm=colors.LogNorm()
                        )
        cax.cla()
        fig.colorbar(img, cax=cax)
    else:
        img.set_data(fg.mass)

    if vel_show:
        fg.pl = FluidGrid.goodiv(fg.pl)
        q.set_UVC(fg.px / fg.pl, fg.py / fg.pl)
    
    tx.set_text('Frame {}'.format(step))
    #print('rendering {} frame...'.format(step))

def animation(step):
    for _ in range(3):
        fg.update()
    render(step)

def main():
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames', default=1_000, type=int, help='set frames')
    parser.add_argument('-iv', '--interval', default=40, type=int, help='set interval between two frames')
    parser.add_argument('-s', '--size', default=49, type=int, help='set size of grid')
    parser.add_argument('-dc', '--dynamic-color', action='store_true', help='set scale of mass dynamin or static')
    parser.add_argument('-vs', '--vel-show', action='store_true', help='set vel show or not')
    parser.add_argument('-a', '--anim', action='store_true')
    args = parser.parse_args()

    frames = args.frames
    interval = args.interval
    siz = args.size
    anim = args.anim

    global dynamic_color, vel_show, fg, img, q
    dynamic_color = args.dynamic_color
    vel_show = args.vel_show

    # fg init here >>>
    fg = FluidGrid.FluidGrid(
        siz, siz, 
        #g=0.0,
        #status_init_func=center33,
        status_update_func=make_spring(),
        )
    # <<< fg init here
    img = ax.imshow(fg.mass,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img, cax=cax)

    if vel_show:
        fg.pl = FluidGrid.goodiv(fg.pl)
        q = ax.quiver(fg.X, fg.Y, fg.px / fg.pl, fg.py / fg.pl, scale=siz * 1.0, pivot='mid')

    # animation
    ani = FuncAnimation(fig, animation, frames=frames, interval=interval)

    if anim:
        FuncAnimation.save(ani, filename="output.mp4")
    else:
        plt.show()

# Your own code begin ->

def thrower(mass, px, py):
    height, width = mass.shape
    my, mx = 2, width // 3
    px[my-1:my+2, mx-1:mx+2] += mass[my-1:my+2, mx-1:mx+2] * 2
    py[my-1:my+2, mx-1:mx+2] += mass[my-1:my+2, mx-1:mx+2] * 2
    return (mass, px, py)

def spin(mass, px, py):
    h, w = mass.shape
    my, mx = h // 2, w // 2
    px[my-h//5, mx] += mass[my-h//2, mx] * 2
    px[my+h//5, mx] -= mass[my+h//2, mx] * 2
    py[my, mx-w//5] -= mass[my, mx-w//2] * 2
    py[my, mx+w//5] += mass[my, mx+w//2] * 2
    return (mass, px, py)

def make_spring(vel=0.0, step=0.05):
    def spring(mass, px, py):
        nonlocal vel
        height, width = mass.shape

        if mass[height // 4 * 1, width // 2] < 1e0:
            py[1, width // 2] = mass[1, width // 2] * vel

        if mass[height // 4 * 1, width // 2] < 1e-1:
            vel += step
        
        if mass[height // 4 * 3, width // 2] > 1e-2:
            vel -= step

        return (mass, px, py)
    
    return spring

def center33(height, width):
    mass = np.zeros((height, width)) + 1e-3
    px = np.zeros((height, width))
    py = np.zeros((height, width))

    mass[height // 2 - 1 : height // 2 + 2 , width // 2 - 1 : width // 2 + 2] = 1

    return (mass, px, py)

# <- Your own code end

if __name__ == '__main__':
    main()