import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solver import *
import os, glob
import cv2

def plot_2D_wave(u, x, xv, y, yv, t, n, step=9):
    if n == 0: # First step, make meshgrid
        global X, Y, count;
        X, Y = np.meshgrid(xv,yv)
        count = step # So that the first step is plotted
    if count == step:
        plt.ioff()
        fig = plt.figure()
        ax  = Axes3D(fig)
        ax.set_zlim3d(-0.1, 0.35)
        ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=plt.cm.gist_earth)
        plt.savefig('figures/2D_wave_%.4i.png' %n, dpi = 150, antialiased=False)
        plt.close()
        count = 0   # Reset counter
    else:
        count += 1


def create_waves(bottom_shape):
    for frame in glob.glob("figures/2D_wave_*.png"):
        os.remove(frame)

    Lx =  1;   Ly = 1
    Nx =  120; Ny = 120
    dt = -1    # Shortcut to maximum timestep
    T  =  0.6; b = 0.8

    def H(x, y, bottom_shape):
        if bottom_shape == 1: # Gaussian
            Bmx = 0.5; Bmy = 0.5; B0 = 1
            Ba = -0.6; Bs = np.sqrt(1./30); b = 0.8
            return B0 + Ba*np.exp( -((x-Bmx)/Bs)**2 - ((y-Bmy)/(b*Bs))**2)

        elif bottom_shape == 2: # Cosine hat
            Bmx = 0.5; Bmy = 0.5; B0 = 1
            Ba = -0.2; Bs = 0.2
            condition = np.sqrt(x**2 + y**2)
            if condition <= Bs and condition >= 0:
                return B0 + Ba*np.cos((np.pi*(x-Bmx))/(2*Bs))*np.cos((np.pi*(y-Bmy))/(2*Bs))
            else:
                return B0

        elif bottom_shape == 3: # Box addition
            Bmx = 0.5; Bmy = 0.5; B0 = 1
            Ba = -0.6; Bs = 0.2; b = 0.8

            if (Bmx-Bs < x < Bmx+Bs) and (Bmy-b*Bs < y < Bmy+b*Bs):
                return B0 + Ba
            else:
                return B0

    def q(x,y): # q = c**2
        g = 9.81 #[m/s^2] acceleration of gravity
        if len(x.shape) == 1:
            q = np.zeros((len(x),len(y)))
            for i,xx in enumerate(x):
                for j,yy in enumerate(y):
                    q[i,j] = g*H(xx,yy,bottom_shape)
        if len(x.shape) == 2:
            q = np.zeros((x.shape[0],y.shape[1]))
            for i,xx in enumerate(x[:,0]):
                for j,yy in enumerate(y[0]):
                    q[i,j] = g*H(xx,yy,bottom_shape)
        return q

    sigma = 0.091; I0 = 0; Im = 0
    Ia = 0.3; Is = np.sqrt(2)*sigma
    def I(x,y):
        return I0 + Ia*np.exp(-((x-Im)/Is)**2)

    f = lambda x,y,t: 0 # Source term
    V = lambda x,y  : 0 # Initial du/dt

    solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=plot_2D_wave, version='vectorized')

def create_video():
    images = []
    for filename in sorted(glob.glob("figures/*")):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        images.append(img)
    out = cv2.VideoWriter('2D_wave_cosine_hat.avi', 0, 4, size)
    for i in images:
        out.write(i)
    out.release()


if __name__ == '__main__':
    import sys
    def raise_error():
        print('\nChoose which wave to simulate:\n')
        print('1: Gaussian hill')
        print('2: Cosine hat hill')
        print('3: Box hill\n')
        sys.exit(1)

    if len(sys.argv) < 2:
        raise_error()

    if sys.argv[1] == '1':
        print("\nRunning gaussian hill. Creating plots in folder 'figures'. \
Use function create_video() afterwards to create movie.")
        create_waves(1)

    if sys.argv[1] == '2':
        print("\nRunning cosine hat hill. Creating plots in folder 'figures'. \
Use function create_video() afterwards to create movie.")
        create_waves(2)

    if sys.argv[1] == '3':
        print("\nRunning box hill. Creating plots in folder 'figures'. \
Use function create_video() afterwards to create movie.")
        create_waves(3)

    #create_video()
