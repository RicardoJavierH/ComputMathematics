#https://scipython.com/
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Acceleration due to gravity, m.s-2 (downwards!).
g = 10
# Circle radius, m.
R = 1
# Time step, s.
dt = 0.001

def solve(u0):
    """Solve the equation of motion for a ball bouncing in a circle.

    u0 = [x0, vx0, y0, vy0] are the initial conditions (position and velocity).

    """

    # Initial time, final time, s.
    t0, tf = 0, 20

    def fun(t, u):
        """Return the derivatives of the dynamics variables packed into u."""
        x, xdot, y, ydot = u
        xddot = 0
        yddot = -g
        return xdot, xddot, ydot, yddot

    def event(t, u):
        """If the ball hits the wall of the circle, trigger the event."""
        return np.hypot(u[0], u[2]) - R*1.01
    # Make sure the event terminates the integration.
    event.terminal = True

    # Keep track of the ball's position in these lists.
    x, y = [], []
    while True:
        # Solve the equations until the ball hits the circular wall or until
        # the time tf.
        soln = solve_ivp(fun, (t0, tf), u0, events=event, dense_output=True)
        if soln.status == 1:
            # We hit the wall: save the path so far...
            tend = soln.t_events[0][0]
            nt = int(tend - t0 / dt) + 1
            tgrid = np.linspace(t0, tend, 100)
            sol = soln.sol(tgrid)
            x.append(sol[0])
            y.append(sol[2])

            # ...and restart the integration with the reflected velocities as
            # the initial conditions.
            u = soln.y[:, -1].copy()
            p = np.array((u[0], u[2]))
            p = p / np.linalg.norm(p)
            v = np.array((u[1], u[3]))
            v = v - 2 * (v @ p) * p
            u0 = p[0], v[0], p[1], v[1]
            t0 = soln.t[-1]
        else:
            # We're done up to tf (or, rather, the last bounce before tf).
            break
    # Concatenate all the paths between bounces together.
    return np.concatenate(x), np.concatenate(y)

# For the animation, set up the path lines and circle patches representing the
# balls.
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
# Make the circular wall a little bit bigger because the ball paths eat into it.
ax.add_patch(plt.Circle((0, 0), 1.02, fc='k', ec='w'))
ax.axis('equal')
ax.axis('off')

line0, = ax.plot([], [])
line1, = ax.plot([], [])

pos = []
u0 = [0.001, 0, 0, 0]
ball0 = ax.add_patch(plt.Circle((u0[0], u0[2]), 0.05, fc='tab:blue', ec='none'))
pos.append(solve(u0))

u0 = [0.0015, 0, 0, 0]
ball1 = ax.add_patch(plt.Circle((u0[0], u0[2]), 0.05, fc='tab:orange', ec='none'))
pos.append(solve(u0))

def init():
    """
    Initialization, because we're blitting and need references to the
    animated objects.
    """
    return line0, line1, ball0, ball1

def animate(i):
    """Draw frame i of the animation."""

    line0.set_data(pos[0][0][:i], pos[0][1][:i])
    ball0.set_center((pos[0][0][i], pos[0][1][i]))

    line1.set_data(pos[1][0][:i], pos[1][1][:i])
    ball1.set_center((pos[1][0][i], pos[1][1][i]))
    return line0, line1, ball0, ball1

interval, nframes = 1000 * dt, int(len(pos[0][0]))
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=False,
                              init_func=init, interval=interval, blit=True)
plt.show()