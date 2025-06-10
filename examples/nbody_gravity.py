import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from cpax.ode.models import hamiltonian_nd
from cpax.ode.integrators import simulate_leapfrog_scan

# Constants
G = 4 * jnp.pi**2

# Masses: Sun, Earth, Mars, Comet
masses = jnp.array([1.0, 3e-6, 3.3e-7, 1e-10])
q0 = jnp.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.5, 0.0],
    [5.0, 0.0],
])
vels = jnp.array([
    [0.0, 0.0],
    [0.0, 2 * jnp.pi],
    [0.0, 1.6 * jnp.pi],
    [0.0, 0.7 * jnp.sqrt(G / 5.0)],
])
p0 = masses[:, None] * vels

# Integrate
dt = 0.001
t_end = 5
n_steps = int(t_end / dt)
f = hamiltonian_nd(masses, potential="gravity")
ts, qs, ps = simulate_leapfrog_scan(q0, p0, t0=0.0, dt=dt, n_steps=n_steps, f=f, masses=masses)

# Energy diagnostics
def kinetic_energy(p, masses):
    return 0.5 * jnp.sum(jnp.sum(p**2, axis=-1) / masses)

def potential_energy(q, masses):
    N = q.shape[0]
    diffs = q[:, None, :] - q[None, :, :]
    distances = jnp.linalg.norm(diffs + jnp.eye(N)[:, :, None], axis=-1)
    mprod = masses[:, None] * masses[None, :]
    mask = 1.0 - jnp.eye(N)
    V_mat = -G * mprod / distances * mask
    return 0.5 * jnp.sum(V_mat)

Es = jnp.array([kinetic_energy(p, masses) + potential_energy(q, masses)
                for q, p in zip(qs, ps)])

# Plot energy
plt.figure()
plt.plot(ts, Es)
plt.xlabel("Time [yr]")
plt.ylabel("Energy [AU²/yr²]")
plt.title("Total Energy over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_conservation.png")
plt.close()

qs = np.array(qs)  # convert to NumPy for plotting
ts = np.array(ts)

colors = ['gold', 'blue', 'orangered', 'gray']
labels = ['Sun', 'Earth', 'Mars', 'Comet']

fig, ax = plt.subplots(figsize=(6,6))
scatters = []
trails   = []
for c, l in zip(colors, labels):
    scat, = ax.plot([], [], 'o', color=c, label=l)
    trail,= ax.plot([], [], '-', color=c, alpha=0.5)
    scatters.append(scat)
    trails.append(trail)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("X [AU]")
ax.set_ylabel("Y [AU]")
ax.set_title("N-body Simulation")
ax.grid(True)
ax.legend()

def init():
    for scat, trail in zip(scatters, trails):
        scat.set_data([], [])
        trail.set_data([], [])
    return scatters + trails

def update(frame):
    for i, (scat, trail) in enumerate(zip(scatters, trails)):
        x, y = qs[frame, i]
        # pass as sequences, even for a single point:
        scat.set_data([x], [y])
        # trail is the full history up to current frame
        xs = qs[:frame+1, i, 0]
        ys = qs[:frame+1, i, 1]
        trail.set_data(xs, ys)
    return scatters + trails

ani = FuncAnimation(fig, update,
                    frames=range(0, len(ts), 20),
                    init_func=init,
                    blit=True,
                    interval=20)

# Use PillowWriter to save a GIF:
writer = PillowWriter(fps=30)
ani.save("nbody_simulation.gif", writer=writer)

plt.close(fig)
print("Animation saved as nbody_simulation.gif")