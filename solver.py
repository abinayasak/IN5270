import numpy as np
import time, sys


def solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
           user_action=None, version='scalar'):
    if version == 'vectorized':
        advance = advance_vectorized
    elif version == 'scalar':
        advance = advance_scalar

    x = np.linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xv = x[:,np.newaxis]          # for vectorized function evaluations
    yv = y[np.newaxis,:]

    c = np.sqrt(q(x,y))
    stability_limit = (1/np.float(c.max()))*(1/np.sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print('error: dt=%g exceeds the stability limit %g' %(dt, stability_limit))

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)    # mesh points in time
    Cx2 = (dt/dx)**2;  Cy2 = (dt/dy)**2    # help variables
    dt2 = dt**2

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: np.zeros((x.shape[0], y.shape[1]))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: np.zeros((x.shape[0], y.shape[1]))


    order = 'C'
    u   = np.zeros((Nx+1,Ny+1), order=order)   # solution array
    u_1 = np.zeros((Nx+1,Ny+1), order=order)   # solution at t-dt
    u_2 = np.zeros((Nx+1,Ny+1), order=order)   # solution at t-2*dt
    f_a = np.zeros((Nx+1,Ny+1), order=order)   # for compiled loops
    V_a = np.zeros((Nx+1,Ny+1), order=order)

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    It = range(0, t.shape[0])


    # Load initial condition into u_1
    if version == 'scalar':
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I(x[i], y[j])
    else: # use vectorized version
        u_1[:,:] = I(xv, yv)

    if user_action is not None:
        user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u = advance_scalar(
            u, u_1, u_2, q, b, f, x, y, t, n,
            Cx2, Cy2, dt2, V, step1=True)

    else:
        f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
        f = f_a[:,:]
        V_a[:,:] = V(xv, yv)  # precompute, size as u
        V = V_a[:,:]
        q_a = np.zeros((Nx+1,Ny+1), order=order)
        q_a[:,:] = q(xv, yv)
        q = q_a[:,:]
        u = advance_vectorized(
            u, u_1, u_2, q, b, f,
            x, y, Cx2, Cy2, dt2, V, step1=True)

    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    u_2, u_1, u = u_1, u, u_2 # Update data structures for next step

    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance(u, u_1, u_2, q, b, f, x, y, t, n, Cx2, Cy2, dt2)
        else:
            u = advance(u, u_1, u_2, q, b, f, x, y, Cx2, Cy2, dt2, V)


        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break

        u_2, u_1, u = u_1, u, u_2  # Update data structures for next step

    u = u_1
    # Important to set u = u_1 if u is to be returned!
    # dt might be computed in this function so return the value
    return dt, u



def advance_scalar(u, u_1, u_2, q, b, f, x, y, t, n, Cx2, Cy2, dt2,
                   V=None, step1=False):
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    dt = np.sqrt(dt2)
    B = 1/(1 + 0.5*b*dt)

    if step1: # Special formula for the firs step (n=0)
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
                + 0.25*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) \
                - (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i,j]-u_1[i-1,j]) ) \
                + 0.25*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) \
                - (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j]-u_1[i,j-1]) ) \
                + 0.5*dt2*f(x[i], y[j], 0)

    else: # General formula for all inner points [1:-1]
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
                + B*0.5*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) \
                - (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i,j]-u_1[i-1,j]) ) \
                + B*0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) \
                - (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j]-u_1[i,j-1]) ) \
                + B*dt2*f(x[i], y[j], t[n])


    if step1: # Boundary condition for the first step
        i = Ix[0] # Boundary condition at x = 0
        for j in Iy[1:-1]:
            u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
            + 0.5*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) ) \
            + 0.25*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j]-u_1[i,j-1]) ) \
            + 0.5*dt2*f(x[i], y[j], 0)

        i = Ix[-1] # Boundary condition at x = Nx
        for j in Iy[1:-1]:
            u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
            + 0.5*Cx2*((q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i-1,j]-u_1[i,j]) ) \
            + 0.25*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j]-u_1[i,j-1]) ) \
            + 0.5*dt2*f(x[i], y[j], 0)

        j = Iy[0] # Boundary condition at y = 0
        for i in Ix[1:-1]:
            u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
            + 0.25*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i,j]-u_1[i-1,j]) ) \
            + 0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) ) \
            + 0.5*dt2*f(x[i], y[j], 0)

        j = Iy[-1] # Boundary condition at y = Nx
        for i in Ix[1:-1]:
            u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
            + 0.25*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i,j]-u_1[i-1,j]) ) \
            + 0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j-1]-u_1[i,j]) ) \
            + 0.5*dt2*f(x[i], y[j], 0)

        i = Ix[0]; j = Iy[0] # Leftmost boundary condition (x = 0 and y = 0)
        u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
        + 0.5*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) ) \
        + 0.5*dt2*f(x[i], y[j], 0)

        i = Ix[-1]; j = Iy[-1] # Rightmost boundary condition (x = Nx and y = Nx)
        u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
        + 0.5*Cx2*( (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i-1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j-1]-u_1[i,j]) ) \
        + 0.5*dt2*f(x[i], y[j], 0)

        i = Ix[0]; j = Iy[-1] # Boundary condition at x = 0 and y = Nx
        u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
        + 0.5*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j-1]-u_1[i,j]) ) \
        + 0.5*dt2*f(x[i], y[j], 0)

        i = Ix[-1]; j = Iy[0] # Boundary condition at x = Nx and y = 0
        u[i,j] = -0.5*b*dt2*V(x[i], y[j]) + dt*V(x[i], y[j]) + u_1[i,j] \
        + 0.5*Cx2*( (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i-1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) ) \
        + 0.5*dt2*f(x[i], y[j], 0)

    else: # Boundary condition for the general steps
        i = Ix[0] # Boundary condition at x = 0
        for j in Iy[1:-1]:
            u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
            + B*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) ) \
            + B*0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j]-u_1[i,j-1]) ) \
            + B*dt2*f(x[i], y[j], t[n])

        i = Ix[-1] # Boundary condition at x = Nx
        for j in Iy[1:-1]:
            u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
            + B*Cx2*( (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i-1,j]-u_1[i,j]) ) \
            + B*0.5*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j]-u_1[i,j-1]) ) \
            + B*dt2*f(x[i], y[j], t[n])

        j = Iy[0] # Boundary condition at y = 0
        for i in Ix[1:-1]:
            u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
            + B*0.5*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i,j]-u_1[i-1,j]) ) \
            + B*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) ) \
            + B*dt2*f(x[i], y[j], t[n])

        j = Iy[-1] # Boundary condition at y = Nx
        for i in Ix[1:-1]:
            u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
            + B*0.5*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) \
            - (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i,j]-u_1[i-1,j]) ) \
            + B*Cy2*( (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j-1]-u_1[i,j]) ) \
            + B*dt2*f(x[i], y[j], t[n])

        i = Ix[0]; j = Iy[0] # Leftmost boundary condition (x = 0 and y = 0)
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) ) \
        + B*dt2*f(x[i], y[j], t[n])

        i = Ix[-1]; j = Iy[-1] # Rightmost boundary condition (x = Nx and y = Nx)
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i-1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j-1]-u_1[i,j]) ) \
        + B*dt2*f(x[i], y[j], t[n])

        i = Ix[0]; j = Iy[-1] # Boundary condition  at x = 0 and y = Nx
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q(x[i], y[j])+q(x[i+1], y[j]))*(u_1[i+1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q(x[i], y[j])+q(x[i], y[j-1]))*(u_1[i,j-1]-u_1[i,j]) ) \
        + B*dt2*f(x[i], y[j], t[n])

        i = Ix[-1]; j = Iy[0] # Boundary condition  at x = Nx and y = 0
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q(x[i], y[j])+q(x[i-1], y[j]))*(u_1[i-1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q(x[i], y[j])+q(x[i], y[j+1]))*(u_1[i,j+1]-u_1[i,j]) ) \
        + B*dt2*f(x[i], y[j], t[n])

    return u

def advance_vectorized(u, u_1, u_2, q, b, f, x, y, Cx2, Cy2, dt2,
                       V, step1=False):
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    dt = np.sqrt(dt2)
    B = 1./(1. + 0.5*b*dt)

    if step1: # Special formula for the firs step (n=0)
        #print('start step1')
        u[1:-1,1:-1] = -0.5*b*dt2*V[1:-1,1:-1] + dt*V[1:-1,1:-1] + u_1[1:-1,1:-1] \
        + 0.25*Cx2*( (q[1:-1,1:-1]+q[2:,1:-1])*(u_1[2:,1:-1]-u_1[1:-1,1:-1]) \
        - (q[1:-1,1:-1]+q[:-2,1:-1])*(u_1[1:-1,1:-1]-u_1[:-2,1:-1]) ) \
        + 0.25*Cy2*( (q[1:-1,1:-1]+q[1:-1,2:])*(u_1[1:-1,2:]-u_1[1:-1,1:-1]) \
        - (q[1:-1,1:-1]+q[1:-1,:-2])*(u_1[1:-1,1:-1]-u_1[1:-1,:-2]) ) \
        + 0.5*dt2*f[1:-1,1:-1]
        #print('end step1')

    else:
        #print('start general step')
        u[1:-1,1:-1] = B*(2*u_1[1:-1,1:-1] + (0.5*b*dt - 1)*u_2[1:-1,1:-1]) \
        + B*0.5*Cx2*( (q[1:-1,1:-1]+q[2:,1:-1])*(u_1[2:,1:-1]-u_1[1:-1,1:-1]) \
        - (q[1:-1,1:-1]+q[:-2,1:-1])*(u_1[1:-1,1:-1]-u_1[:-2,1:-1]) ) \
        + B*0.5*Cy2*( (q[1:-1,1:-1]+q[1:-1,2:])*(u_1[1:-1,2:]-u_1[1:-1,1:-1]) \
        - (q[1:-1,1:-1]+q[1:-1,:-2])*(u_1[1:-1,1:-1]-u_1[1:-1,:-2]) ) \
        + B*dt2*f[1:-1,1:-1]
        #print('end general step')

    if step1: # Boundary condition for the first step
        #print('start step1 boundaries')
        i = Ix[0] # Boundary condition at x = 0
        u[i,1:-1] = -0.5*b*dt2*V[i,1:-1] + dt*V[i,1:-1] + u_1[i,1:-1] \
        + 0.5*Cx2*( (q[i,1:-1]+q[i+1,1:-1])*(u_1[i+1,1:-1]-u_1[i,1:-1]) ) \
        + 0.25*Cy2*( (q[i,1:-1]+q[i,2:])*(u_1[i,2:]-u_1[i,1:-1]) \
        - (q[i,1:-1]+q[i,:-2])*(u_1[i,1:-1]-u_1[i,:-2]) ) \
        + 0.5*dt2*f[i,1:-1]

        i = Ix[-1] # Boundary condition at x = Nx
        u[i,1:-1] = -0.5*b*dt2*V[i,1:-1] + dt*V[i,1:-1] + u_1[i,1:-1] \
        + 0.5*Cx2*( (q[i,1:-1]+q[i-1,1:-1] )*(u_1[i-1,1:-1]-u_1[i,1:-1]) ) \
        + 0.25*Cy2*( (q[i,1:-1]+q[i,2:])*(u_1[i,2:]-u_1[i,1:-1]) \
        - (q[i,1:-1]+q[i,:-2])*(u_1[i,1:-1]-u_1[i,:-2]) ) \
        + 0.5*dt2*f[i,1:-1]

        j = Iy[0] # Boundary condition at y = 0
        u[1:-1,j] = -0.5*b*dt2*V[1:-1,j] + dt*V[1:-1,j] + u_1[1:-1,j] \
        + 0.25*Cx2*( ( q[1:-1,j]+q[2:,j] )*(u_1[2:,j]-u_1[1:-1,j]) \
        - (q[1:-1,j]+q[:-2,j])*(u_1[1:-1,j]-u_1[:-2,j]) ) \
        + 0.5*Cy2*( (q[1:-1,j]+q[1:-1,j+1])*(u_1[1:-1,j+1]-u_1[1:-1,j]) ) \
        + 0.5*dt2*f[1:-1,j]

        j = Iy[-1] # Boundary condition at y = Ny
        u[1:-1,j] = -0.5*b*dt2*V[1:-1,j] + dt*V[1:-1,j] + u_1[1:-1,j] \
        + 0.25*Cx2*( (q[1:-1,j]+q[2:,j])*(u_1[2:,j]-u_1[1:-1,j]) \
        - (q[1:-1,j]+q[:-2,j])*(u_1[1:-1,j]-u_1[:-2,j]) ) \
        + 0.5*Cy2*( (q[1:-1,j]+q[1:-1,j-1])*(u_1[1:-1,j-1]-u_1[1:-1,j]) ) \
        + 0.5*dt2*f[1:-1,j]

        i = Ix[0]; j = Iy[0] # Leftmost boundary condition (x = 0 and y = 0)
        u[i,j] = -0.5*b*dt2*V[i,j] + dt*V[i,j] + u_1[i,j] \
        + 0.5*Cx2*( (q[i,j]+q[i+1,j])*(u_1[i+1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q[i,j]+q[i,j+1])*(u_1[i,j+1]-u_1[i,j]) ) \
        + 0.5*dt2*f[i,j]

        i = Ix[-1]; j = Iy[-1] # Rightmost boundary condition (x = Nx and y = Ny)
        u[i,j] = -0.5*b*dt2*V[i,j] + dt*V[i,j] + u_1[i,j] \
        + 0.5*Cx2*( (q[i,j]+q[i-1,j])*(u_1[i-1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q[i,j]+q[i,j-1])*(u_1[i,j-1]-u_1[i,j]) ) \
        + 0.5*dt2*f[i,j]

        i = Ix[0]; j = Iy[-1] # Boundary condition at x = 0 and y = Ny
        u[i,j] = -0.5*b*dt2*V[i,j] + dt*V[i,j] + u_1[i,j] \
        + 0.5*Cx2*( (q[i,j]+q[i+1,j])*(u_1[i+1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q[i,j]+q[i,j-1])*(u_1[i,j-1]-u_1[i,j]) ) \
        + 0.5*dt2*f[i,j]

        i = Ix[-1]; j = Iy[0] # Boundary condition at x = Nx and y = 0
        u[i,j] = -0.5*b*dt2*V[i,j] + dt*V[i,j] + u_1[i,j] \
        + 0.5*Cx2*( (q[i,j]+q[i-1,j])*(u_1[i-1,j]-u_1[i,j]) ) \
        + 0.5*Cy2*( (q[i,j]+q[i,j+1])*(u_1[i,j+1]-u_1[i,j]) ) \
        + 0.5*dt2*f[i,j]
        #print('end step1 boundaries')

    else: # Boundary condition for the general steps
        i = Ix[0] # Boundary condition at x = 0
        u[i,1:-1] = B*(2*u_1[i,1:-1] + (0.5*b*dt - 1)*u_2[i,1:-1]) \
        + B*Cx2*( (q[i,1:-1]+q[i+1,1:-1])*(u_1[i+1,1:-1]-u_1[i,1:-1]) ) \
        + B*0.5*Cy2*( (q[i,1:-1]+q[i,2:])*(u_1[i,2:]-u_1[i,1:-1]) \
        - (q[i,1:-1]+q[i,:-2])*(u_1[i,1:-1]-u_1[i,:-2]) ) \
        + B*dt2*f[i,1:-1]

        i = Ix[-1] # Boundary condition at x = Nx
        u[i,1:-1] = B*(2*u_1[i,1:-1] + (0.5*b*dt - 1)*u_2[i,1:-1]) \
        + B*Cx2*( (q[i,1:-1]+q[i-1,1:-1])*(u_1[i-1,1:-1]-u_1[i,1:-1]) ) \
        + B*0.5*Cy2*( (q[i,1:-1]+q[i,2:])*(u_1[i,2:]-u_1[i,1:-1]) \
        - (q[i,1:-1]+q[i,:-2])*(u_1[i,1:-1]-u_1[i,:-2]) ) \
        + B*dt2*f[i,1:-1]

        j = Iy[0] # Boundary condition at y = 0
        u[1:-1,j] = B*(2*u_1[1:-1,j] + (0.5*b*dt - 1)*u_2[1:-1,j]) \
        + B*0.5*Cx2*( (q[1:-1,j]+q[2:,j])*(u_1[2:,j]-u_1[1:-1,j]) \
        - (q[1:-1,j]+q[:-2,j])*(u_1[1:-1,j]-u_1[:-2,j]) ) \
        + B*Cy2*( (q[1:-1,j]+q[1:-1,j+1])*(u_1[1:-1,j+1]-u_1[1:-1,j]) ) \
        + B*dt2*f[1:-1,j]

        j = Iy[-1] # Boundary condition at y = Ny
        u[1:-1,j] = B*(2*u_1[1:-1,j] + (0.5*b*dt - 1)*u_2[1:-1,j]) \
        + B*0.5*Cx2*( (q[1:-1,j]+q[2:,j])*(u_1[2:,j]-u_1[1:-1,j]) \
        - (q[1:-1,j]+q[:-2,j])*(u_1[1:-1,j]-u_1[:-2,j]) ) \
        + B*Cy2*( (q[1:-1,j]+q[1:-1,j-1])*(u_1[1:-1,j-1]-u_1[1:-1,j]) ) \
        + B*dt2*f[1:-1,j]

        i = Ix[0]; j = Iy[0] # Leftmost boundary condition (x = 0 and y = 0)
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q[i,j]+q[i+1,j])*(u_1[i+1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q[i,j]+q[i,j+1])*(u_1[i,j+1]-u_1[i,j]) ) \
        + B*dt2*f[i,j]

        i = Ix[-1]; j = Iy[-1] # Rightmost boundary condition (x = Nx and y = Ny)
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q[i,j]+q[i-1,j])*(u_1[i-1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q[i,j]+q[i,j-1])*(u_1[i,j-1]-u_1[i,j]) ) \
        + B*dt2*f[i,j]

        i = Ix[0]; j = Iy[-1] # Boundary condition  at x = 0 and y = Ny
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q[i,j]+q[i+1,j])*(u_1[i+1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q[i,j]+q[i,j-1])*(u_1[i,j-1]-u_1[i,j]) ) \
        + B*dt2*f[i,j]

        i = Ix[-1]; j = Iy[0] # Boundary condition  at x = Nx and y = 0
        u[i,j] = B*(2*u_1[i,j] + (0.5*b*dt - 1)*u_2[i,j]) \
        + B*Cx2*( (q[i,j]+q[i-1,j])*(u_1[i-1,j]-u_1[i,j]) ) \
        + B*Cy2*( (q[i,j]+q[i,j+1])*(u_1[i,j+1]-u_1[i,j]) ) \
        + B*dt2*f[i,j]

    return u
