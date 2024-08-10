import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class FluidGrid:
    @staticmethod
    def diffusion(m, dr, pc):
        rm = (m * (1-dr)
        + np.vstack((m[1:, ], m[-1, ])) * dr/4
        + np.vstack((m[0, ], m[:-1, ])) * dr/4
        + np.hstack((m[:, 1:], m[:, -1].reshape(-1, 1))) * dr/4
        + np.hstack((m[:, 0].reshape(-1, 1), m[:, :-1])) * dr/4
        )

        dpx = (-np.sqrt(np.hstack((m[:, 1:], m[:, -1].reshape(-1, 1))) * dr/4)
        + np.sqrt(np.hstack((m[:, 0].reshape(-1, 1), m[:, :-1])) * dr/4)
        ) * pc

        dpy = (-np.sqrt(np.vstack((m[1:, ], m[-1, ])) * dr/4)
        + np.sqrt(np.vstack((m[0, ], m[:-1, ])) * dr/4)
        ) * pc

        return (rm, dpx, dpy)
        
    @staticmethod
    def mechanical_motion(m, px, py, X, Y, dt):
        dx = px / m * dt
        dy = py / m * dt

        # symbols of dx, dy
        sdx = (dx > 0) * 2 - 1
        sdx[:, 0] *= sdx[:, 0] >= 0
        sdx[:, -1] *= sdx[:, -1] <= 0
        sdy = (dy > 0) * 2 - 1
        sdy[0, ] *= sdy[0, ] >= 0
        sdy[-1, ] *= sdy[-1, ] <= 0
        
        # abs of dx, dy
        adx = np.minimum(np.abs(dx), np.ones_like(dx))
        ady = np.minimum(np.abs(dy), np.ones_like(dy))
        
        # proportion of each part
        p00 = (1 - adx) * (1 - ady)
        px0 = adx * (1 - ady)
        p0y = (1 - adx) * ady
        pxy = adx * ady
        
        # calculate results 
        rm = m * p00
        rpx = px * p00
        rpy = py * p00
        # very bad undefined behavior
        '''
        rm[X + sdx, Y] += m * px0
        rpx[X + sdx, Y] += px * px0
        rpy[X + sdx, Y] += py * px0
        rm[X, Y + sdy] += m * p0y
        rpx[X, Y + sdy] += px * p0y
        rpy[X, Y + sdy] += py * p0y
        rm[X + sdx, Y + sdy] += m * pxy
        rpx[X + sdx, Y + sdy] += px * pxy
        rpy[X + sdx, Y + sdy] += py * pxy
        '''
        def fuck(a, y, x, c):
            yy = y.reshape(-1)
            xx = x.reshape(-1)
            cc = c.reshape(-1)
            for i in range(len(yy)):
                a[yy[i], xx[i]] += cc[i]
        fuck(rm, Y + sdy, X, m * p0y)
        fuck(rpx, Y + sdy, X, px * p0y)
        fuck(rpy, Y + sdy, X, py * p0y)
        fuck(rm, Y, X + sdx, m * px0)
        fuck(rpx, Y, X + sdx, px * px0)
        fuck(rpy, Y, X + sdx, py * px0)
        fuck(rm, Y + sdy, X + sdx, m * pxy)
        fuck(rpx, Y + sdy, X + sdx, px * pxy)
        fuck(rpy, Y + sdy, X + sdx, py * pxy)

        # boundary
        rpx[:, 0] *= rpx[:, 0] > 0
        rpx[:, -1] *= rpx[:, -1] <0
        rpy[0, ] *= rpy[0, ] > 0
        rpy[-1, ] *= rpy[-1, ] < 0

        # done at last...
        return (rm, rpx, rpy)

    def __init__(self, height, width, dr=0.4, pc=0.5, vk=1.0, dt=0.2):
        self.height = height
        self.width = width
        self.diffusion_rate = dr
        self.pressC = pc
        self.vel_keep = vk
        self.mmdt = dt
        self.X, self.Y = np.meshgrid(np.arange(width), np.arange(height))

        self.mass = np.abs(np.random.randn(height, width))
        self.px = np.zeros((height, width)) * self.mass
        self.py = np.zeros((height, width)) * self.mass
        
        # exp
        self.mass = np.zeros_like(self.mass)
        self.mass += 1e-3
        self.mass[height//2, width//2] = 1
        self.px = np.zeros_like(self.px)
        self.py = np.zeros_like(self.py)

        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)

    def debug(self, hint):
        return
        print(hint, self.mass, self.px, self.py, '\n', sep='\n')

    def update(self, dr=None, pc=None, dt=None, vk=None):
        self.debug('0')
        # diffusion
        if dr == None:
            dr = self.diffusion_rate
        if pc == None:
            pc = self.pressC
        self.mass, dpx, dpy = FluidGrid.diffusion(self.mass, dr, pc)
        self.px += dpx
        self.py += dpy

        self.debug('diff')
        # mechanical motion
        if dt == None:
            dt = self.mmdt
        self.mass, self.px, self.py = FluidGrid.mechanical_motion(
            self.mass, self.px, self.py, self.X, self.Y, dt
        )

        self.debug('mech')
        # vel loss
        if vk == None:
            vk = self.vel_keep
        self.px *= vk
        self.py *= vk
        
        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)

        self.debug('vel loss')
        print('mass: {:.2f}, sum_pl: {:.2f}, uniformity: {:.2f}'.format(np.sum(self.mass), np.sum(self.pl), np.linalg.norm(self.mass)))

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
    cax.cla()

    if dynamic_color:
        global img
        img = ax.imshow(fg.mass,
                        cmap='coolwarm',
                        origin='lower',
                        norm=colors.LogNorm()
                        )
    else:
        img.set_data(fg.mass)
    
    fig.colorbar(img, cax=cax)

    if vel_show:
        q.set_UVC(fg.py / (fg.pl + 1e-8), fg.px / (fg.pl + 1e-8), fg.pl)
    
    tx.set_text('Frame {}'.format(step))
    print('rendering {} frame...'.format(step))

def animation(step):
    fg.update()
    render(step)

import argparse

def main():
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames', default=1_000, type=int, help='set frames')
    parser.add_argument('-iv', '--interval', default=40, type=int, help='set interval between two frames')
    parser.add_argument('-s', '--size', default=20, type=int, help='set size of grid')
    parser.add_argument('-sc', '--static-color', action="store_true", help='set scale of mass dynamin or static')
    parser.add_argument('-v', '--vel', action="store_false", help='set vel show or not')
    args = parser.parse_args()

    frames = args.frames
    interval = args.interval
    siz = args.size

    global fg, img, q, dynamic_color, vel_show
    fg = FluidGrid(siz, siz)
    img = ax.imshow(fg.mass,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img, cax=cax)
    dynamic_color = not args.static_color
    vel_show = args.vel
    if vel_show:
        q = ax.quiver(fg.X, fg.Y, fg.py / (fg.pl + 1e-8), fg.px / (fg.pl + 1e-8), fg.pl, scale=siz * 2.5)

    # animation
    anim = FuncAnimation(fig, animation, frames=frames, interval=interval)
    # FuncAnimation.save(anim, filename="output.mp4")
    plt.show()

if __name__ == '__main__':
    main()