import numpy as np

def goodiv(x):
    return np.maximum(x, 1e-8)

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
    m = goodiv(m)
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
    rpx += (-np.sqrt(np.hstack((m[:, 1:], m[:, -1].reshape(-1, 1))) * dr)
    + np.sqrt(np.hstack((m[:, 0].reshape(-1, 1), m[:, :-1])) * dr)
    ) * pc

    rpy += (-np.sqrt(np.vstack((m[1:, ], m[-1, ])) * dr)
    + np.sqrt(np.vstack((m[0, ], m[:-1, ])) * dr)
    ) * pc

    return (rm, rpx, rpy)

def mechanical_motion(m, px, py, X, Y, dt):
    m = goodiv(m)
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
    adx = np.abs(dx) 
    ady = np.abs(dy)
    k = np.maximum(np.maximum(adx, ady), 1)
    adx /= k
    ady /= k

    # proportion of each part
    p00 = (1 - adx) * (1 - ady)
    px0 = adx * (1 - ady)
    p0y = (1 - adx) * ady
    pxy = adx * ady
    
    # calculate results 
    rm = m * p00
    rpx = px * p00
    rpy = py * p00
    # very bad undefined behavior, made my program slower (maybe)
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

def randn_status_init(height, width):
    mass = np.abs(np.random.randn(height, width))
    px = np.random.randn(height, width) * mass
    py = np.random.randn(height, width) * mass
    
    return (mass, px, py)

class FluidGrid:
    def __init__(self, height, width, dr=0.025, pc=0.2, vk=0.99, g=0.2, dt=0.2, 
                 status_init_func=randn_status_init, status_update_func=None):
        self.height = height
        self.width = width
        self.diffusion_rate = dr
        self.pressC = pc
        self.vel_keep = vk
        self.g = g
        self.mmdt = dt
        self.X, self.Y = np.meshgrid(np.arange(width), np.arange(height))

        self.mass, self.px, self.py = status_init_func(height, width)
        self.status_update_func = status_update_func
        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)

    #def debug(self, hint):
    #    print(hint, self.mass, self.px, self.py, '\n', sep='\n')

    def update(self, dr=None, pc=None, g=None, dt=None, vk=None):
        # diffusion
        if dr == None:
            dr = self.diffusion_rate
        if pc == None:
            pc = self.pressC
        self.mass, self.px, self.py = diffusion(self.mass, self.px, self.py, dr, pc)

        # gravity
        if g == None:
            g = self.g
        if dt == None:
            dt = self.mmdt
        self.py -= self.mass * g * dt

        # personal status update
        if self.status_update_func != None:
            self.mass, self.px, self.py = self.status_update_func(self.mass, self.px, self.py)

        # vel loss
        if vk == None:
            vk = self.vel_keep
        self.px *= vk
        self.py *= vk
        
        # mechanical motion
        self.mass, self.px, self.py = mechanical_motion(
            self.mass, self.px, self.py, self.X, self.Y, dt
        )

        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)
        #print('mass: {:.2f}, sum_pl: {:.2f}, uniformity: {:.2f}'.format(np.sum(self.mass), np.sum(self.pl), np.linalg.norm(self.mass)))