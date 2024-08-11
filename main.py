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
        q.set_UVC(fg.px / fg.pl, fg.py / fg.pl, fg.pl)
    
    tx.set_text('Frame {}'.format(step))
    #print('rendering {} frame...'.format(step))

def animation(step):
    for _ in range(2):
        fg.update()
    render(step)

def main():
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames', default=1_000, type=int, help='set frames')
    parser.add_argument('-iv', '--interval', default=40, type=int, help='set interval between two frames')
    parser.add_argument('-s', '--size', default=20, type=int, help='set size of grid')
    parser.add_argument('-sc', '--static-color', action='store_true', help='set scale of mass dynamin or static')
    parser.add_argument('-nv', '--non-vel', action='store_true', help='set vel show or not')
    parser.add_argument('-a', '--anim', action='store_true')
    args = parser.parse_args()

    frames = args.frames
    interval = args.interval
    siz = args.size
    anim = args.anim

    global dynamic_color, vel_show, fg, img, q
    dynamic_color = not args.static_color
    vel_show = not args.non_vel

    # fg init here!
    fg = FluidGrid.FluidGrid(siz, siz, status_update_func=make_spring())

    img = ax.imshow(fg.mass,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img, cax=cax)

    if vel_show:
        fg.pl = FluidGrid.goodiv(fg.pl)
        q = ax.quiver(fg.X, fg.Y, fg.px / fg.pl, fg.py / fg.pl, fg.pl, scale=siz * 2.5)

    # animation
    ani = FuncAnimation(fig, animation, frames=frames, interval=interval)

    if anim:
        FuncAnimation.save(ani, filename="output.mp4")
    else:
        plt.show()

# Your own code begin ->

def make_spring(acc=0):
    def spring(mass, px, py):
        nonlocal acc
        height, width = mass.shape

        if py[height // 4, width // 2] <= 0:
            py[0, width // 2] += acc
            acc += 1
        elif acc > 0:
            acc -= 1

        return (mass, px, py)
    
    return spring

def spring2(mass, px, py):
    height, width = mass.shape
    
    py[0, width // 4] += 4
    px[0, width // 4] += 1

    return (mass, px, py)

def center33(height, width):
    mass = np.zeros((height, width)) + 1e-3
    px = np.zeros((height, width))
    py = np.zeros((height, width))

    mass[height // 2 - 1 : height // 2 + 2 , width // 2 - 1 : width // 2 + 2] = 1

    return (mass, px, py)

# Your own code end

if __name__ == '__main__':
    main()