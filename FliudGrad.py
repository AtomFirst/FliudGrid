import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.autolayout'] = True
fig, ax = plt.subplots(1,1)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
tx = ax.set_title('start')
img = None
q = None

fg = None

class FluidGrid:
    @staticmethod
    def diffusion(z, k):
        return (z * (1-k)
        + np.vstack((z[1:, ], z[-1, ])) * k/4
        + np.vstack((z[0, ], z[:-1, ])) * k/4
        + np.hstack((z[:, 1:], z[:, -1].reshape(-1, 1))) * k/4
        + np.hstack((z[:, 0].reshape(-1, 1), z[:, :-1])) * k/4
        )

    @staticmethod
    def solvePress(m, k):
        Fx = (np.hstack((m[:, 0].reshape(-1, 1), m[:, :-1])) 
        - np.hstack((m[:, 1:], m[:, -1].reshape(-1, 1)))
        ) * k
        Fy = (np.vstack((m[0, ], m[:-1, ])) 
        - np.vstack((m[1:, ], m[-1, ]))
        ) * k

        return (Fx, Fy)
        
    @staticmethod
    def mechanical_motion(m, px, py, X, Y, vk, dt):
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
        def fuck(a, x, y, c):
            xx = x.reshape(-1)
            yy = y.reshape(-1)
            cc = c.reshape(-1)
            for i in range(len(xx)):
                a[xx[i], yy[i]] += cc[i]
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
        fuck(rm, X + sdx, Y, m * px0)
        fuck(rpx, X + sdx, Y, px * px0)
        fuck(rpy, X + sdx, Y, py * px0)
        fuck(rm, X, Y + sdy, m * p0y)
        fuck(rpx, X, Y + sdy, px * p0y)
        fuck(rpy, X, Y + sdy, py * p0y)
        fuck(rm, X + sdx, Y + sdy, m * pxy)
        fuck(rpx, X + sdx, Y + sdy, px * pxy)
        fuck(rpy, X + sdx, Y + sdy, py * pxy)

        # boundary
        rpx[:, 0] *= rpx[:, 0] > 0
        rpx[:, -1] *= rpx[:, -1] <0
        rpy[0, ] *= rpy[0, ] > 0
        rpy[-1, ] *= rpy[-1, ] < 0
        
        # vel loss
        rpx *= vk
        rpy *= vk

        # done at last...
        return (rm, rpx, rpy)

    def __init__(self, height, width, dr=0.1, pc=3.0, vk=0.3, dt=0.02):
        self.height = height
        self.width = width
        self.diffusion_rate = dr
        
        self.pressC = pc
        self.vel_keep = vk
        self.mmdt = dt

        self.X, self.Y = np.meshgrid(np.arange(width), np.arange(height))
        self.mass = np.abs(np.random.randn(height, width))
        
        self.px = np.random.randn(height, width) * self.mass * 0.2
        self.py = np.random.randn(height, width) * self.mass * 0.2
        self.color = np.sqrt(self.px ** 2 + self.py ** 2)

    def update(self, k=None, dt=None):
        if k == None:
            k = self.pressC

        Fx, Fy = FluidGrid.solvePress(self.mass, k)

        if dt == None:
            dt = self.mmdt

        self.px += Fx * dt
        self.py += Fy * dt

        self.mass, self.px, self.py = FluidGrid.mechanical_motion(
            self.mass, self.px, self.py, self.X, self.Y, self.vel_keep, dt
        )
        
        self.color = np.sqrt(self.px ** 2 + self.py ** 2)
        print('mass: {:.2f}, Ev: {:.2f}, qwq: {:.2f}'.format(np.sum(self.mass), np.sum(self.color ** 2), np.linalg.norm(self.mass)))

def render(step):
    global img
    cax.cla()
    img = ax.imshow(fg.mass,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img, cax=cax)
    q.set_UVC(fg.px, fg.py)
    tx.set_text('Frame {}'.format(step))
    print('rendering {} frame...'.format(step))

def update(step):
    fg.update()
    return render(step)

import argparse

def main():
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames', default=1_000, type=int, help='set frames')
    parser.add_argument('-iv', '--interval', default=40, type=int, help='set interval')
    parser.add_argument('-s', '--size', default=20, type=int, help='set size')
    args = parser.parse_args()

    frames = args.frames
    interval = args.interval
    siz = args.size

    global fg, img, q
    fg = FluidGrid(siz, siz)
    img = ax.imshow(fg.mass,
                    cmap='coolwarm',
                    origin='lower',
                    norm=colors.LogNorm()
                    )
    fig.colorbar(img, cax=cax)
    q = ax.quiver(fg.X, fg.Y, fg.px, fg.py, fg.color)

    # animation
    anim = FuncAnimation(fig, update, frames=frames, interval=interval)
    # FuncAnimation.save(anim, filename="output.mp4")
    plt.show()

if __name__ == '__main__':
    main()