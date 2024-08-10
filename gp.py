import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.autolayout'] = True
fig = plt.figure()
ax = fig.add_subplot(111)
tx = None
frames = None
interval = None
div = None
cax = None
img = None

siz = None
x = None
y = None
X,Y = None,None
z = None

def calc(step, k=0.1,g=0.1):
    global z
    # version 2
    z = (z * (1-k)
    + np.vstack((z[1:, ], z[-1, ])) * k/4
    + np.vstack((z[0, ], z[:siz-1, ])) * k/4
    + np.hstack((z[:, 1:], z[:, -1].reshape(-1, 1))) * k/4
    + np.hstack((z[:, 0].reshape(-1, 1), z[:, :siz-1])) * k/4
    )

    t= z[1:, ].copy()
    z[1:, ] *= 1 - g
    z[:siz-1] += t * g

def render(step):
    global cax, img, fig, tx, ax
    cax.cla()
    img = ax.imshow(z,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img, cax=cax)
    tx.set_text('Frame {}'.format(step))
    print('rendering {} frame...'.format(step))
    #plt.savefig('{}.jpg'.format(step))

# animation

def update(step):
    global z
    calc(step)
    return render(step)

import argparse

def main():
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames', default=1_000, type=int, help='set frames')
    parser.add_argument('-iv', '--interval', default=40, type=int, help='set interval')
    parser.add_argument('-s', '--size', default=20, type=int, help='set size')
    args = parser.parse_args()

    global frames, interval, siz, x, y, X, Y, z, ax, div, cax, tx
    frames = args.frames
    interval = args.interval
    siz = args.size

    x = np.linspace(-1, 1, siz)
    y = np.linspace(-1, 1, siz)
    X,Y = np.meshgrid(x,y)
    z = np.random.randn(siz, siz)
    z = np.abs(z)

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    tx = ax.set_title('Frame {}'.format(0))

    # show
    ani = anim(fig, update, frames=frames, interval=interval)

    #anim.save(ani, filename="output.mp4")
    
    plt.show()

if __name__ == '__main__':
    main()