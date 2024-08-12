import numpy as np

def goodiv(x, minv=1e-8):
    return np.maximum(x, minv)

def diffusion(m, px, py, E, dr, pc):
    # mass transition
    dmx = np.diff(m, axis=1) * dr
    dmy = np.diff(m, axis=0) * dr
    rm = m.copy()
    rm[:, :-1] += dmx
    rm[:, 1:] -= dmx
    rm[:-1, ] += dmy
    rm[1:, ] -= dmy
    
    # momentum transition 
    dpxx = np.diff(px, axis=1) * dr
    dpxy = np.diff(px, axis=0) * dr
    rpx = px.copy()
    rpx[:, :-1] += dpxx
    rpx[:, 1:] -= dpxx
    rpx[:-1, ] += dpxy
    rpx[1:, ] -= dpxy

    dpyx = np.diff(py, axis=1) * dr
    dpyy = np.diff(py, axis=0) * dr
    rpy = py.copy()
    rpy[:, :-1] += dpyx
    rpy[:, 1:] -= dpyx
    rpy[:-1, ] += dpyy
    rpy[1:, ] -= dpyy

    # energy transition
    dEx = np.diff(E, axis=1) * dr
    dEy = np.diff(E, axis=0) * dr
    rE = E.copy()
    rE[:, :-1] += dEx
    rE[:, 1:] -= dEx
    rE[:-1, ] += dEy
    rE[1:, ] -= dEy

    # pressure do work
    k = np.sqrt(dr) * pc
    rpx[:, :-1] -= np.sqrt(E[:, 1:]) * k
    rpx[:, -1] -= np.sqrt(E[:, -1]) * k
    rpx[:, 0] += np.sqrt(E[:, 0]) * k
    rpx[:, 1:] += np.sqrt(E[:, :-1]) * k

    rpy[:-1, ] -= np.sqrt(E[1:, ]) * k
    rpy[-1, ] -= np.sqrt(E[-1, ]) * k
    rpy[0, ] += np.sqrt(E[0, ]) * k
    rpy[1:, ] += np.sqrt(E[:-1, ]) * k
    '''
    # version without sqrt
    rpx[:, :-1] -= E[:, 1:] * k
    rpx[:, -1] -= E[:, -1] * k
    rpx[:, 0] += E[:, 0] * k
    rpx[:, 1:] += E[:, :-1] * k

    rpy[:-1, ] -= E[1:, ] * k
    rpy[-1, ] -= E[-1, ] * k
    rpy[0, ] += E[0, ] * k
    rpy[1:, ] += E[:-1, ] * k
    '''

    return (rm, rpx, rpy, rE)

def mechanical_motion(m, px, py, E, X, Y, dt):
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
    # if dx, dy > 1
    if True:
        # angle keep
        k = np.maximum(np.maximum(adx, ady), 1)
        adx /= k
        ady /= k
    else:
        # angle not keep
        adx = np.minimum(adx, 1)
        ady = np.minimum(ady, 1)

    # proportion of each part
    p00 = (1 - adx) * (1 - ady)
    px0 = adx * (1 - ady)
    p0y = (1 - adx) * ady
    pxy = adx * ady
    
    # calculate results 
    rm = m * p00
    rpx = px * p00
    rpy = py * p00
    rE = E * p00
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
    add_to(rE, Y + sdy, X, E * p0y)
    add_to(rm, Y, X + sdx, m * px0)
    add_to(rpx, Y, X + sdx, px * px0)
    add_to(rpy, Y, X + sdx, py * px0)
    add_to(rE, Y, X + sdx, E * px0)
    add_to(rm, Y + sdy, X + sdx, m * pxy)
    add_to(rpx, Y + sdy, X + sdx, px * pxy)
    add_to(rpy, Y + sdy, X + sdx, py * pxy)
    add_to(rE, Y + sdy, X + sdx, E * pxy)

    # boundary
    rpx[:, 0] *= rpx[:, 0] > 0
    rpx[:, -1] *= rpx[:, -1] <0
    rpy[0, ] *= rpy[0, ] > 0
    rpy[-1, ] *= rpy[-1, ] < 0
    
    # done at last...
    return (rm, rpx, rpy, rE)

def randn_status_init(height, width):
    mass = goodiv(np.random.randn(height, width) * 0.1 + 1.0)
    px = np.random.randn(height, width) * mass
    py = np.random.randn(height, width) * mass
    E = goodiv((np.random.randn(height, width) * 0.1 + 1.0)) * mass

    return (mass, px, py, E)

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

        self.mass, self.px, self.py, self.E = status_init_func(height, width)
        self.status_update_func = status_update_func
        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)

    #def debug(self, hint):
    #    print(hint, self.mass, self.px, self.py, '\n', sep='\n')

    def update(self, dr=None, pc=None, g=None, dt=None, vk=None, mu=None):
        # diffusion
        if dr == None:
            dr = self.diffusion_rate
        if pc == None:
            pc = self.pressC
        self.mass, self.px, self.py, self.E = diffusion(self.mass, self.px, self.py, self.E, dr, pc)

        # gravity
        if g == None:
            g = self.g
        if dt == None:
            dt = self.mmdt
        self.py -= self.mass * g * dt

        # personal status update
        if self.status_update_func != None:
            self.mass, self.px, self.py, self.E = self.status_update_func(self.mass, self.px, self.py, self.E)

        # vel loss
        if vk == None:
            vk = self.vel_keep
        self.px *= vk
        self.py *= vk
        
        # mechanical motion
        self.mass, self.px, self.py, self.E = mechanical_motion(
            self.mass, self.px, self.py, self.E, self.X, self.Y, dt
        )

        self.pl = np.sqrt(self.px ** 2 + self.py ** 2)
        #print('mass: {:.2f}, sum_pl: {:.2f}, uniformity: {:.2f}'.format(np.sum(self.mass), np.sum(self.pl), np.linalg.norm(self.mass)))