import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class FluidGrid:
    @staticmethod
    def goodiv(x):
        return np.maximum(x, 1e-8)

    @staticmethod
    def diffusion(m, px, py, dr, pc):
        dmx = np.diff(m, axis=1) * dr
        dmy = np.diff(m, axis=0) * dr

        # mass transition
        rm = m.copy()
        rm[:, :-1] += dmx
        rm[:, 1:] -= dmx
        rm[:-1, ] += dmy
        rm[1:, ] -= dmy

        # momentum transition 
        rpx = px.copy()
        # r2l means moving from right cells to left cells
        m = FluidGrid.goodiv(m)
        dpxr2l = (dmx > 0) * dmx / m[:, 1:] * px[:, 1:]
        dpxl2r = (dmx < 0) * (-dmx) / m[:, :-1] * px[:, :-1]
        rpx[:, :-1] += dpxr2l - dpxl2r
        rpx[:, 1:] += dpxl2r - dpxr2l
        rpy = py.copy()
        dpyu2d = (dmy > 0) * dmy / m[1:, ] * px[1:, ]
        dpyd2u = (dmy < 0) * (-dmy) / m[:-1, ] * px[:-1, ]
        rpy[:-1, ] += dpyu2d - dpyd2u
        rpy[1:, ] += dpyd2u -dpyu2d

        # pressure do work
        rpx += (-np.sqrt(np.hstack((m[:, 1:], m[:, -1].reshape(-1, 1))) * dr/4)
        + np.sqrt(np.hstack((m[:, 0].reshape(-1, 1), m[:, :-1])) * dr/4)
        ) * pc

        rpy += (-np.sqrt(np.vstack((m[1:, ], m[-1, ])) * dr/4)
        + np.sqrt(np.vstack((m[0, ], m[:-1, ])) * dr/4)
        ) * pc

        return (rm, rpx, rpy)
        
    @staticmethod
    def mechanical_motion(m, px, py, X, Y, dt):
        m = FluidGrid.goodiv(m)
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
        def add_to(a, y, x, c):
            yy = y.reshape(-1)
            xx = x.reshape(-1)
            cc = c.reshape(-1)
            for i in range(len(yy)):
                a[yy[i], xx[i]] += cc[i]
        add_to(rm, Y + sdy, X, m * p0y)
        add_to(rpx, Y + sdy, X, px * p0y)
        add_to(rpy, Y + sdy, X, py * p0y)
        add_to(rm, Y, X + sdx, m * px0)
        add_to(rpx, Y, X + sdx, px * px0)
        add_to(rpy, Y, X + sdx, py * px0)
        add_to(rm, Y + sdy, X + sdx, m * pxy)
        add_to(rpx, Y + sdy, X + sdx, px * pxy)
        add_to(rpy, Y + sdy, X + sdx, py * pxy)

        # boundary
        rpx[:, 0] *= rpx[:, 0] > 0
        rpx[:, -1] *= rpx[:, -1] <0
        rpy[0, ] *= rpy[0, ] > 0
        rpy[-1, ] *= rpy[-1, ] < 0

        # done at last...
        return (rm, rpx, rpy)

    def __init__(self, height, width, dr=0.025, pc=0.2, vk=0.99, g=0.2, dt=0.2):
        self.height = height
        self.width = width
        self.diffusion_rate = dr
        self.pressC = pc
        self.vel_keep = vk
        self.g = g
        self.mmdt = dt
        self.X, self.Y = np.meshgrid(np.arange(width), np.arange(height))

        self.mass = np.abs(np.random.randn(height, width))
        self.px = np.zeros((height, width)) * self.mass
        self.py = np.zeros((height, width)) * self.mass
        
        # exp
        self.dd = 0
        '''
        self.mass = np.zeros_like(self.mass)
        self.mass += 1e-3
        self.mass[height // 2 - 1 : height // 2 + 2 , width // 2 - 1 : height // 2 + 2] = 1
        self.px = np.zeros_like(self.px)
        self.py = np.zeros_like(self.py)
        '''
        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)

    def debug(self, hint):
        return
        print(hint, self.mass, self.px, self.py, '\n', sep='\n')

    def update(self, dr=None, pc=None, g=None, dt=None, vk=None):
        # diffusion
        if dr == None:
            dr = self.diffusion_rate
        if pc == None:
            pc = self.pressC
        self.mass, self.px, self.py = FluidGrid.diffusion(self.mass, self.px, self.py, dr, pc)

        # gravity
        if g == None:
            g = self.g
        if dt == None:
            dt = self.mmdt
        self.py -= self.mass * g * dt

        # mechanical motion
        self.mass, self.px, self.py = FluidGrid.mechanical_motion(
            self.mass, self.px, self.py, self.X, self.Y, dt
        )

        # vel loss
        if vk == None:
            vk = self.vel_keep
        self.px *= vk
        self.py *= vk
        
        # spring
        if self.py[self.height // 2, self.width // 2] <= 0:
            self.py[0, self.width // 2] += self.dd
            self.dd += 1
        elif self.dd > 0:
            self.dd -= 1

        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)
        #print('mass: {:.2f}, sum_pl: {:.2f}, uniformity: {:.2f}'.format(np.sum(self.mass), np.sum(self.pl), np.linalg.norm(self.mass)))

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
        global img
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

import argparse

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

    fg = FluidGrid(siz, siz)
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

if __name__ == '__main__':
    main()