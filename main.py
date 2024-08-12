import argparse
import numpy as np
import FluidGrid
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = [10, 5]
fig, [ax1, ax2] = plt.subplots(1,2)
div1, div2 = make_axes_locatable(ax1), make_axes_locatable(ax2)
cax1, cax2 = div1.append_axes('right', '5%', '5%'), div2.append_axes('right', '5%', '5%')
tx1, tx2 = ax1.set_title('pressure'), ax2.set_title('temperature')

img1, img2 = None, None
q = None
dynamic_color = True
vel_show = True
fg = None

def render(step):
    global fg
    fg.mass = FluidGrid.goodiv(fg.mass)
    if dynamic_color:
        global img1, img2
        img1 = ax1.imshow(fg.E,
                        cmap='coolwarm',
                        origin='lower',
                        norm=colors.LogNorm()
                        )
        img2 = ax2.imshow(fg.E / fg.mass,
                        cmap='coolwarm',
                        origin='lower',
                        norm=colors.LogNorm()
                        )
        cax1.cla()
        cax2.cla()
        fig.colorbar(img1, cax=cax1)
        fig.colorbar(img2, cax=cax2)
    else:
        img1.set_data(fg.mass)
        img2.set_data(fg.E / fg.mass)

    if vel_show:
        fg.pl = FluidGrid.goodiv(fg.pl)
        q.set_UVC(fg.px / fg.pl, fg.py / fg.pl)
    
    tx1.set_text('Frame {} pressure'.format(step))
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

    global dynamic_color, vel_show, fg, img1, img2, q
    dynamic_color = args.dynamic_color
    vel_show = args.vel_show

    # fg init here >>>
    fg = FluidGrid.FluidGrid(
        siz, siz, 
        #pc=0.0,
        #g=0.0,
        status_init_func=center33,
        status_update_func=heater,
        )
    # <<< fg init here
    
    img1 = ax1.imshow(fg.E,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img1, cax=cax1)
    
    img2 = ax2.imshow(fg.E / fg.mass,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img2, cax=cax2)

    if vel_show:
        fg.pl = FluidGrid.goodiv(fg.pl)
        q = ax1.quiver(fg.X, fg.Y, fg.px / fg.pl, fg.py / fg.pl, scale=siz * 1.0, pivot='mid')

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

def heater(mass, px, py, E):
    h, w = mass.shape
    mx1, mx2 = w//3, w//3*2
    E[:3, mx1-1:mx1+2] *= 1.01
    E[:3, mx2-1:mx2+2] *= 0.95

    return (mass, px, py, E)

def make_spring(vel=0.0, step=0.05):
    def spring(mass, px, py, E):
        nonlocal vel
        height, width = mass.shape

        if mass[height // 4 * 1, width // 2] < 1e0:
            py[1, width // 2] = mass[1, width // 2] * vel

        if mass[height // 4 * 1, width // 2] < 1e-1:
            vel += step
        
        if mass[height // 4 * 3, width // 2] > 1e-2:
            vel -= step

        return (mass, px, py, E)
    
    return spring

def center33(height, width):
    mass = np.zeros((height, width)) + 1e-3
    px = np.zeros((height, width))
    py = np.zeros((height, width))

    mass[height // 2 - 1 : height // 2 + 2 , width // 2 - 1 : width // 2 + 2] = 1
    
    E = (np.random.randn(height, width) * 0.1 + 1.0) * mass
    
    return (mass, px, py, E)

# <- Your own code end

if __name__ == '__main__':
    main()