from solver import *

def constant(Nx, Ny, version):
    """Exact discrete solution of the scheme."""

    def exact_solution(x, y, t): return 5
    def I(x, y): return 5
    def V(x, y): return 0
    def f(x, y, t): return 0
    def q(x,y): return 1

    Lx = 5;  Ly = 2
    b = 1.5; dt = -1 # use longest possible steps
    T = 4

    def assert_no_error(u, x, xv, y, yv, t, n):
        u_e = exact_solution(xv, yv, t[n])
        diff = abs(u - u_e).max()
        error.append(diff)
        tol = 1E-12
        msg = 'diff=%g, step %d, time=%g' % (diff, n, t[n])
        assert diff < tol, msg

    new_dt, u = solver(
        I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=assert_no_error, version=version)
    print(u)

def test_constant():
    # Test a series of meshes where Nx > Ny and Nx < Ny
    global error
    versions = 'scalar', 'vectorized'
    for Nx in range(4, 6, 2):
        for Ny in range(4, 6, 2):
            for version in versions:
                error = []
                print('testing', version, 'for %dx%d mesh' % (Nx, Ny))
                constant(Nx, Ny, version)
                print('Largest error: %.2e' % max(error))


def test_plug_wave():
    def V(x, y): return 0
    def f(x, y, t): return 0
    def q(x,y): return 1
    def c(x,y): return np.sqrt(q(x,y))

    Lx = 1.1; Ly = 1.1
    Nx = 11;  Ny = 11
    dt = 0.1; T = 1.1
    b  = 0

    class Action:
        """Store last solution."""
        def __call__(self, u, x, xv, y, yv, t, n):
            if n == len(t)-1:
                self.u = u.copy(); self.x = x.copy()
                self.y = y.copy(); self.t = t[n]

    action = Action()

    def Ix_scalar(x,y): # Scalar initial condition
        if abs(x-Lx/2.0) > 0.1: return 0
        else: return 1


    def Ix_vec(x,y): # vectorized initial condition
        I = np.zeros(x.shape)
        for i in range(len(x[:,0])):
            if abs(x[i,0]-Lx/2.0) > 0.1:
                I[i,0] = 0
            else:
                I[i,0] = 1
        return I


    def Iy_scalar(x,y): # For scalar scheme
        if abs(y-Ly/2.0) > 0.1: return 0
        else: return 1


    def Iy_vec(x,y):
        I = np.zeros(y.shape)
        for j in range(len(y[0,:])):
            if abs(y[0,j]-Ly/2.0) > 0.1:
                I[0,j] = 0
            else:
                I[0,j] = 1
        return I


    solver(Ix_scalar, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=action, version='scalar')
    u_s_x = action.u

    solver(Iy_scalar, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=action, version='scalar')
    u_s_y = action.u

    solver(Ix_vec, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=action, version='vectorized')
    u_v_x = action.u

    solver(Iy_vec, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
        user_action=action, version='vectorized')
    u_v_y = action.u

    diff1 = abs(u_s_x - u_v_x).max()
    diff2 = abs(u_s_y - u_v_y).max()

    tol = 1e-13
    assert diff1 < tol and diff1 < tol

    u_0_x = np.zeros((Nx+1,Ny+1))
    u_0_x[:,:] = Ix_vec(action.x[:,np.newaxis], action.y[np.newaxis,:])

    u_0_y = np.zeros((Nx+1,Ny+1))
    u_0_y[:,:] = Iy_vec(action.x[:,np.newaxis], action.y[np.newaxis,:])

    def check_wave():
        #Just to check that the wave splits for different timestep (in x-direction)
        #At which timestep to plot is modified in Action() by changing 'len(t)-1'
        import matplotlib.pyplot as plt
        x = np.linspace(0, Lx, Nx+1)
        y = np.linspace(0, Ly, Ny+1)
        plt.plot(x, u_v_x)
        plt.title("t=%.2f" % action.t)
        plt.xlabel('Lx')
        plt.ylabel('U')
        plt.show()
    #check_wave()

    diff3 = abs(u_s_x - u_0_x).max()
    diff4 = abs(u_s_y - u_0_y).max()
    diff5 = abs(u_v_x - u_0_x).max()
    diff6 = abs(u_v_y - u_0_y).max()
    assert diff3 < tol and diff4 < tol and diff5 < tol and diff6 < tol

class Error:
    #Stores the error norm
    def __init__(self, u_e):
        self.u_e = u_e
        self.E = 0
        self.h = 0

    def __call__(self, u, x, xv, y, yv, t, n):
        if n == 0:
            self.h = t[1] - t[0]
        if n == len(t)-1:
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            e = self.u_e(xv,yv,t[n]) - u
            self.E = np.sqrt(dx*self.h*np.sum(e**2))


def test_undamped_waves():
    Lx = 1; Ly = 1
    dt = -1;  T = 0.4
    b  = 0
    mx = 2; my = 3
    kx = (mx*np.pi)/Lx
    ky = (my*np.pi)/Ly
    w = np.sqrt(kx**2 + ky**2); A = 1.5

    def u_e(x, y, t): return A*np.cos(kx*x)*np.cos(ky*y)*np.cos(w*t)
    def I(x, y): return A*np.cos(kx*x)*np.cos(ky*y)
    def V(x, y): return 0
    def f(x, y, t): return 0
    def q(x,y): return 1
    def c(x,y): return np.sqrt(q(x,y))

    N_values = [5,10,20,40,80,160,320]
    E_values = []   # Contains largest error for different Nx values
    h_values = []
    error = Error(u_e)

    for N in N_values: # Run all experiments
        print('Running Nx=Ny:', N)
        Nx = Ny = N
        dt = -1

        solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
            user_action=error, version='vectorized')

        E_values.append(error.E)
        h_values.append(error.h)

    r = [(np.log(E_values[i]/E_values[i-1]))/np.log(h_values[i]/h_values[i-1]) for i in range(1, len(N_values))]


    print("\n            CONVERGENCE RATES WITH DETAILS            ")
    print(" N(i) | N(i+1) |   dt(i)   |  dt(i+1)  |  r(i)  | \n")
    for i in range(len(N_values)-1):
        print(" %-3i      %-4i     %-9.3E   %-9.3E   %-5.4f" \
            %(N_values[i], N_values[i+1], h_values[i], h_values[i+1], r[i]))
    print('')


def source_term(A_value, B_value, kx_value, ky_value, w_value, b_value):
    import sympy as sym
    x, y, t, b, A, B, w, c, kx, ky, L = sym.symbols('x, y, t, b, A, B, w, c, kx, ky, L')

    q = lambda x,y: 1
    c_value = 1
    u_e = lambda x,y,t: (A*sym.cos(w*t) + B*sym.sin(w*t))*(sym.exp(-c*t))*sym.cos(kx*x)*sym.cos(ky*y)

    utt = sym.diff(u_e(x,y,t),t,t)
    ut = sym.diff(u_e(x,y,t),t)
    ux = sym.diff(u_e(x,y,t),x)
    dxqux = sym.diff(q(x,y)*ux,x)
    uy = sym.diff(u_e(x,y,t),y)
    dyquy = sym.diff(q(x,y)*uy,y)

    f_sym  = sym.simplify(utt+b*ut-dxqux-dyquy) # Source term as a symolic function
    f_sym = f_sym.subs(A,A_value).subs(B,B_value).subs(b,b_value).subs(w,w_value).subs(kx,kx_value)\
    .subs(ky,ky_value).subs(c, c_value)
    f_func = sym.lambdify((x, y, t), f_sym, modules='numpy')
    return f_func

def test_manufactured_solution():
    Lx = 1; Ly = 1
    dt = -1;  T = 0.4
    b  = 4
    mx = 4; my = 2
    kx = (mx*np.pi)/Lx
    ky = (my*np.pi)/Ly
    w = np.sqrt(kx**2 + ky**2);
    A = 1.5; B = 3

    def u_e(x, y, t): return (A*np.cos(w*t) + B*np.sin(w*t))*(np.exp(-c(x,y)*t))*np.cos(kx*x)*np.cos(ky*y)
    def I(x, y): return A*np.cos(kx*x)*np.cos(ky*y)
    def V(x, y): return (w*B - c(x,y)*A)*np.cos(kx*x)*np.cos(ky*y)
    def q(x,y): return 1
    def c(x,y): return np.sqrt(q(x,y))
    f = source_term(A, B, kx, ky, w, b)

    N_values = [5,10,20,40,80,160,320]
    E_values = []   # Contains largest error for different Nx values
    h_values = []
    error = Error(u_e)

    for N in N_values: # Run all experiments
        print('Running Nx=Ny:', N)
        Nx = Ny = N
        dt = -1

        solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T,
            user_action=error, version='vectorized')

        E_values.append(error.E)
        h_values.append(error.h)

    r = [(np.log(E_values[i]/E_values[i-1]))/np.log(h_values[i]/h_values[i-1]) for i in range(1, len(N_values))]


    print("\n            CONVERGENCE RATES WITH DETAILS            ")
    print(" N(i) | N(i+1) |   dt(i)   |  dt(i+1)  |  r(i)  | \n")
    for i in range(len(N_values)-1):
        print(" %-3i      %-4i     %-9.3E   %-9.3E   %-5.4f" \
            %(N_values[i], N_values[i+1], h_values[i], h_values[i+1], r[i]))
    print('')

    tol = 0.05
    assert r[-1]-2 < tol and r[-1]-2 > 0  #Test to check that r converges to two

if __name__ == '__main__':
    print('\nTesting constant u: \n')
    test_constant()
    print('\nTesting plug wave: \n')
    test_plug_wave()
    print('\nTesting convergence rates of undamped waves: \n')
    test_undamped_waves()
    print('Testing convergence rates for manufactured solution \n')
    test_manufactured_solution()
    
    #To run nosetest write: $nosetests -v tests.py
