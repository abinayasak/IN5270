import numpy as np
import matplotlib.pyplot as plt
import sys

def solver(q, f, I, U_0, U_L, L, dt, dx, T, C, V, user_action, version, task, boundary_condition):
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    dt2 = dt*dt                       # Help variable in the scheme
    dx2 = dx*dx                       # Help variable in the scheme
    C2 = dt2/dx2                      # Help variable in the scheme

    # Wrap user-given V if None or 0
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)


    if task == 'd':
        x = np.linspace(-dx/2, L+dx/2, Nx+1)       # Mesh points in space
        u = np.zeros(Nx+1)     # Solution array at new time level
        u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
        u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back

        Ix = np.arange(1, u.shape[0]-1)
        It = range(0, Nt+1)
        # Load initial condition into u_1
        for i in Ix:
            u_1[i] = I(x[i])

        # Insert Neumann boundary condition to leftmost and rightmost values
        i = Ix[0]
        u_1[i-1] = u_1[i+1]
        i = Ix[-1]
        u_1[i+1] = u_1[i-1]

        if user_action is not None:
            user_action(u_1[1:-1], x[1:-1], t, 0)  # Returning u_1 that consist physical solution, that is u_1[1:-1] and x[1:-1]

        # Special formula for the first step
        for i in Ix:
            u[i] = u_1[i] + dt*V(x[i]) + \
                0.5*C2*(0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i]) - \
                0.5*(q(x[i]) + q(x[i-1]))*(u_1[i] - u_1[i-1])) + \
                0.5*dt2*f(x[i], t[0])

        i = Ix[0]  # Set boundary condition for the leftmost point
        u[i] = u_1[i] + dt*V(x[i]) + C2*(0.5*(q(x[i])+q(x[i+1]))*(u_1[i+1]-u_1[i])) + dt2*f(x[i], t[0])
        i = Ix[-1]  # Set boundary condition for the rightmost point
        u[i] = u_1[i] + dt*V(x[i]) - C2*(0.5*(q(x[i])+q(x[i-1]))*(u_1[i]-u_1[i-1])) + dt2*f(x[i], t[0])

        # Insert Neumann boundary condition to points outside the domain
        i = Ix[0]
        u[i-1] = u[i+1]
        i = Ix[-1]
        u[i+1] = u[i-1]

        if user_action is not None:
            user_action(u[1:-1], x[1:-1], t, 1) # Returning u_1 that consist physical solution, that is u_1[1:-1] and x[1:-1]


    if task == 'a' or task == 'b' or task == 'c':
        u = np.zeros(Nx+1)     # Solution array at new time level
        u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
        u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back

        Ix = range(0, Nx+1)
        It = range(0, Nt+1)

        # Load initial condition into u_1
        for i in Ix:
            u_1[i] = I(x[i])

        if user_action is not None:
            user_action(u_1, x, t, 0)

        # Special formula for the first step (eq. 50 modified, using u[n-1] = u[n+1] - 2*dt*V)
        for i in Ix[1:-1]:
            u[i] = u_1[i] + dt*V(x[i]) + \
                0.5*C2*(0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i])  - \
                    0.5*(q(x[i]) + q(x[i-1]))*(u_1[i] - u_1[i-1])) + \
                0.5*dt2*f(x[i], t[0])

        if boundary_condition == 'neumann_a':

            i = Ix[0]  # Set boundary condition for the leftmost point
            if U_0 is None:
                u[i] = u_1[i] + dt*V(x[i]) + C2*q(x[i])*(u_1[i+1] - u_1[i]) + 0.5*dt2*f(x[i], t[0])
            else:
                u[0] = U_0(dt)

            i = Ix[-1]  # Set boundary condition for the rightmost point
            if U_L is None:
                u[i] = u_1[i] + dt*V(x[i]) + C2*q(x[i])*(u_1[i-1] - u_1[i]) + 0.5*dt2*f(x[i], t[0])
            else:
                u[i] = U_L(dt)

        elif boundary_condition == 'neumann_b':

            i = Ix[0]  # Set boundary condition for the leftmost point
            if U_0 is None:
                u[i] = u_1[i] + dt*V(x[i]) + C2*(0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i])) + \
                0.5*dt2*f(x[i], t[0])
            else:
                u[0] = U_0(dt)

            i = Ix[-1]
            if U_L is None:  # Set boundary condition for the rightmost point
                u[i] = u_1[i] + dt*V(x[i]) + C2*(0.5*(q(x[i]) + q(x[i-1]))*(u_1[i-1] - u_1[i])) + \
                0.5*dt2*f(x[i], t[0])
            else:
                u[i] = U_L(dt)

        elif boundary_condition == 'onesided_difference': #Using one sided difference method

            i = Ix[0]  # Set boundary condition for the leftmost point
            if U_0 is None:
                u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i])) + \
                0.5*dt2*f(x[i], t[0])
            else:
                u[0] = U_0(dt)

            i = Ix[-1]  # Set boundary condition for the rightmost point
            if U_L is None:
                u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(0.5*(q(x[i]) + q(x[i-1]))*(u_1[i] - u_1[i-1])) + \
                0.5*dt2*f(x[i], t[0])
            else:
                u[i] = U_L(dt)

    if user_action is not None:
        user_action(u, x, t, 1)

    u_2, u_1, u = u_1, u, u_2  # Update data structures for next step


    if task == 'd':
        for n in range(1,Nt):
            # Update all inner points
            if version == 'vectorized':
                u[1:-1] = -u_2[1:-1] + 2*u_1[1:-1] + \
                  C2*(0.5*(q(x[1:-1]) + q(x[2:]))*(u_1[2:] - u_1[1:-1]) -
                      0.5*(q(x[1:-1]) + q(x[:-2]))*(u_1[1:-1] - u_1[:-2])) + \
                  dt2*f(x[1:-1], t[n])
            else:
                raise ValueError('version=%s' % version)

            i = Ix[0]  # Set boundary condition for the leftmost point
            u[i] = -u_2[i] + 2*u_1[i] + C2*(0.5*(q(x[i])+q(x[i+1]))*(u_1[i+1]-u_1[i])) + dt2*f(x[i], t[n])
            i = Ix[-1]  # Set boundary condition for the rightmost point
            u[i] = -u_2[i] + 2*u_1[i] - C2*(0.5*(q(x[i])+q(x[i-1]))*(u_1[i]-u_1[i-1])) + dt2*f(x[i], t[n])

            # Insert Neumann boundary condition to points outside the domain
            i = Ix[0]
            u[i-1] = u[i+1]
            i = Ix[-1]
            u[i+1] = u[i-1]

            if user_action is not None:
                if user_action(u[1:-1], x[1:-1], t, n+1): # Returning u_1 that consist physical solution, that is u_1[1:-1] and x[1:-1]
                    break

            u_2, u_1, u = u_1, u, u_2  # Update data structures for next step


    if task == 'a' or task == 'b' or task == 'c':
        for n in It[1:-1]:
            # Update all inner points
            if version == 'vectorized':

                u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
              C2*(0.5*(q(x[1:-1]) + q(x[2:]))*(u_1[2:] - u_1[1:-1]) -
                  0.5*(q(x[1:-1]) + q(x[:-2]))*(u_1[1:-1] - u_1[:-2])) + \
              dt2*f(x[1:-1], t[n])
            else:
                raise ValueError('version=%s' % version)

            # Insert boundary conditions
            if boundary_condition == 'neumann_a':

                i = Ix[0]  # Set boundary condition for the leftmost point
                if U_0 is None:
                    u[i] = -u_2[i] + 2*u_1[i] + C2*2*q(x[i])*(u_1[i+1]-u_1[i]) + dt2*f(x[i], t[n])
                else:
                    u[0] = U_0(t[n+1])

                i = Ix[-1]
                if U_L is None:
                    # Set boundary condition for the rightmost point
                    u[i] = -u_2[i] + 2*u_1[i] + C2*2*q(x[i])*(u_1[i-1]-u_1[i]) + dt2*f(x[i], t[n])
                else:
                    u[i] = U_L(t[n+1])


            elif boundary_condition == 'neumann_b':

                i = Ix[0]  # Set boundary condition for the leftmost point
                if U_0 is None:
                    u[i] = -u_2[i] + 2*u_1[i] + 2*C2*(0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i])) + \
                    dt2*f(x[i], t[n])

                else:
                    u[0] = U_0(dt)

                i = Ix[-1]  # Set boundary condition for the rightmost point
                if U_L is None:
                    u[i] = -u_2[i] + 2*u_1[i] + 2*C2*(0.5*(q(x[i]) + q(x[i-1]))*(u_1[i-1] - u_1[i])) + \
                    dt2*f(x[i], t[n])
                else:
                    u[i] = U_L(dt)

            elif boundary_condition == 'onesided_difference': #Using one sided difference method

                i = Ix[0]  # Set boundary condition for the leftmost point
                if U_0 is None:
                    u[i] = -u_2[i] + 2*u_1[i] + C2*(0.5*(q(x[i]) + q(x[i+1]))*(u_1[i+1] - u_1[i])) + \
                    dt2*f(x[i], t[n])
                else:
                    u[0] = U_0(dt)

                i = Ix[-1]  # Set boundary condition for the rightmost point
                if U_L is None:
                    u[i] = -u_2[i] + 2*u_1[i] + C2*(-0.5*(q(x[i]) + q(x[i-1]))*(u_1[i] - u_1[i-1])) + \
                    dt2*f(x[i], t[n])
                else:
                    u[i] = U_L(dt)

            if user_action is not None:
                if user_action(u, x, t, n+1):
                    break

            u_2, u_1, u = u_1, u, u_2  # Update data structures for next step

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    return u, x, t


def source_term(L_value, which_q):
    import sympy as sym
    x, t, L = sym.symbols('x t L')

    #Choosing q accordingly to task a or b
    if which_q == 'a':
        q = lambda x: 1 + (x-(L/2))**4
    elif which_q == 'b':
        q = lambda x: 1 + sym.cos(np.pi*x/L)

    u = lambda x,t: sym.cos(sym.pi*x/L)*sym.cos(t)  #Setting w=1

    utt = sym.diff(u(x,t),t,t)
    ux = sym.diff(u(x,t),x)
    dxqux = sym.diff(q(x)*ux,x)

    f_sym  = sym.simplify(utt-dxqux).subs(L,L_value) # Source term as a symolic function
    return sym.lambdify((x, t), f_sym, modules='numpy') # Source term in as a non-symbolic function

def error(u, x, t, n):
    # Finds the L2-norm error.
    L = 1.0
    u_e = np.cos(np.pi*x/L)*np.cos(t[n])  # Setting w=1 and calculating the exact solution
    e = u_e - u
    E = np.sqrt(dx*sum(e**2))
    E_list.append(E.max()) # Finding the largest error and appending it to a list


def I(x):
    return np.cos(np.pi*x/L) # Initial condition

def q(x):
    if which_q == 'a':
        return 1 + (x-L/2)**4
    if which_q == 'b':
        return 1 + np.cos(np.pi*x/L)


def raise_error():
    print('\nChoose which task to run (a, b, c or d)')
    print('If running task c or d, add which q value to run it with by adding a \
(for q from task a) or b (for q from task b) behind the task')
    print("\nUsage for task a and b: python Neumann_discr.py a")
    print("Usage for task c and d: python Neumann_discr.py c a\n")
    sys.exit(1)

if len(sys.argv) < 2:
    print(len(sys.argv))
    raise_error()

if sys.argv[1] == 'a':
    which_q = 'a'
    task = 'a'
    condition = 'neumann_a' # neumann_a: dudx = 0

if sys.argv[1] == 'b':
    which_q = 'b'
    task = 'b'
    condition = 'neumann_b' # neumann_b: dudx = 0 and dqdx = 0.

if sys.argv[1] == 'c':
    if len(sys.argv) != 3:
        raise_error()
    else:
        which_q = sys.argv[2]
        task = 'c'
        condition = 'onesided_difference'

if sys.argv[1] == 'd':
    if len(sys.argv) != 3:
        raise_error()
    else:
        which_q = sys.argv[2]
        task = 'd'
        condition = None



T = 3.0 # Temporal domain
L = 1.0 # Spatial domain
C = 1.0 # Courant number
f = source_term(L_value=L, which_q=which_q) # Using sympy to find source term
V = None
U_0 = None
U_L = None

Nx_values = range(50,800+1,50)
E_values = []   # Contains largest error for different Nx values
dx_values = []  # Contains corresponding dx value

for Nxi in Nx_values: # Run all experiments
    print('Running Nx:', Nxi)
    dx = float(L)/Nxi
    dt = C*dx/np.sqrt(q(0))
    E_list = []

    u, x, t = solver(q, f, I, U_0, U_L, L, dt, dx, T, C, V,
                                user_action=error, version='vectorized', task=task, boundary_condition=condition)

    dx_values.append(dx)
    E_values.append(max(E_list))


r = [np.log(E_values[i-1]/E_values[i])/np.log(dx_values[i-1]/dx_values[i]) for i in range(1, len(Nx_values))]


print("\n            CONVERGENCE RATES WITH DETAILS            ")
print(" Nx(i) | Nx(i+1) |   dx(i)   |  dx(i+1)  |  r(i)  | \n")
for i in range(len(Nx_values)-1):
    print(" %-3i      %-4i     %-9.3E   %-9.3E   %-5.4f" \
        %(Nx_values[i], Nx_values[i+1], dx_values[i], dx_values[i+1], r[i]))
print('')


def plot_convergence():
    plt.figure()
    plt.plot(Nx_values[1:], r)
    plt.xlabel('Nx')
    plt.ylabel('r')
    plt.title('Convergence Rates')
    plt.show()

plot_convergence()
