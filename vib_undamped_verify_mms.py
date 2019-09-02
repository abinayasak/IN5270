import sympy as sym
import numpy as np

V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u, dt) + w**2*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = u(t+dt) - I + 0.5*dt**2*w**2*I - dt*V - 0.5*dt**2*f.subs(t, 0)
    R = R.subs(t, 0)
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt) - 2*u(t) + u(t-dt))/dt**2

def solver(V, I, w, dt, f, T):
    """
    Central finite difference method is used to solve
    u'' + w**2*u = f for t=(0,T], u(0)=I and u'(0)=V, with timestep dt.
    """

    Nt = int(round(T/dt))             # round() returns a floating number of the nearest integer. int() makes the value a integer.
    u = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)

    u[0] = I
    u[1] = u[0] - 0.5*dt**2*w**2*I + dt*V + 0.5*dt**2*f(t[0])
    for n in range(1, Nt):
        u[n+1] = (2-dt**2*w**2)*u[n] - u[n-1] + dt**2*f(t[n])
    return u, t

def test_quadratic_function():
    """
    Comparing the numerial solution with the exact solution.
    """
    global b, V, I, w, f, t
    b, V, I, w = 4, 0.5, 1, 1.7
    u_e = lambda t: b*t**2 + V*t + I  # uses b, V, I, w as numbers with t as a variable
    f = ode_source_term(u_e)          # adds u_e to the source term
    f = sym.lambdify(t, f)            # makes a function out of f that you can calculate on for different t using f(t).
    dt = 2./w                         # choosing the largest timestep possible
    u, t = solver(I=I, V=V, f=f, w=w, dt=dt, T=5) # calculating the numerical solution
    u_e = u_e(t)                      # calculating the exact solution for same t values used to calculate the numerical u
    error = np.abs(u - u_e).max()     # finding the largest error in the mesh
    tol = 1e-9
    assert error < tol
    print('Error in numerical solution:', error)


def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print('=== Testing exact solution: %s ===' % u)
    print("Initial conditions u(0)=%s, u'(0)=%s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0)))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print('residual step1:', residual_discrete_eq_step1(u))
    print('residual:', residual_discrete_eq(u))

def linear():
    """Testing linear function on the form: V*t + I."""
    print("Testing linear function")
    main(lambda t: V*t + I)

def quadratic():
    """Testing quadratic function on the form: b*t**2 + V*t + I."""
    print("Testing quadratic function")
    b = sym.symbols('b') #Arbitrary constant
    main(lambda t: b*t**2 + V*t + I)

def cubic():
    """Testing cubic function on the form: a*t**3 + b*t**2 + V*t + I."""
    print("Testing cubic function")
    a, b = sym.symbols('a, b') #Arbitrary constants
    main(lambda t: a*t**3 + b*t**2 + V*t + I)

if __name__ == '__main__':
    linear()
    quadratic()
    cubic()
    test_quadratic_function()
