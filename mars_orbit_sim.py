import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import astropy.units as u
from astropy.constants import G, M_sun
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
AU = 1.496e11 * u.m  # Astronomical Unit
MU = (G * M_sun).to(u.m**3 / u.s**2).value  # Gravitational parameter for Sun

# Initial conditions
r_earth = 1.0 * AU.value
r_mars_initial = 1.52 * AU.value
r_mars_target = 1.2 * AU.value

v_earth = np.sqrt(MU / r_earth)
v_mars_initial = np.sqrt(MU / r_mars_initial)
v_mars_target = np.sqrt(MU / r_mars_target)

# State vectors [x, y, vx, vy] assuming start at (r, 0) with velocity (0, v)
state_earth = [r_earth, 0.0, 0.0, v_earth]
state_mars = [r_mars_initial, 0.0, 0.0, v_mars_initial]

# Hohmann transfer calculations
r1 = r_mars_initial
r2 = r_mars_target
dv1 = np.sqrt(MU / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)  # Negative for retrograde
dv2 = np.sqrt(MU / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))  # Negative for retrograde
transfer_time = np.pi * np.sqrt((r1 + r2)**3 / (8 * MU))

print(f"Delta v1 (retrograde): {dv1:.2f} m/s")
print(f"Delta v2 (retrograde): {dv2:.2f} m/s")
print(f"Transfer time: {transfer_time / (365.25 * 24 * 3600):.2f} years")

# Equations of motion for central gravity
def orbit_eqs(t, y, mu=MU):
    x, y_pos, vx, vy = y  # Renamed y to y_pos to avoid conflict
    r = np.sqrt(x**2 + y_pos**2)
    ax = -mu * x / r**3
    ay = -mu * y_pos / r**3
    return [vx, vy, ax, ay]

# Simulate Earth's full orbit (for reference, ~2 Mars years)
t_full = 2 * 2 * np.pi * np.sqrt(r1**3 / MU)
sol_earth = solve_ivp(orbit_eqs, [0, t_full], state_earth, rtol=1e-8)

# Apply dv1 to Mars for transfer (add to vy, since dv1 < 0)
state_transfer = state_mars.copy()
state_transfer[3] += dv1

# Integrate transfer ellipse
sol_transfer = solve_ivp(orbit_eqs, [0, transfer_time], state_transfer, rtol=1e-8)

# At perihelion, apply dv2: get direction and set to circular orbit for accuracy
last_pos = sol_transfer.y[0:2, -1]
last_r = np.linalg.norm(last_pos)
tang_dir = np.array([-last_pos[1], last_pos[0]]) / last_r  # CCW tangential
v_circ = np.sqrt(MU / last_r)
state_final = np.concatenate((last_pos, v_circ * tang_dir))

# Integrate final orbit
t_final_span = [transfer_time, transfer_time + (t_full - transfer_time)]
sol_final = solve_ivp(orbit_eqs, t_final_span, state_final, rtol=1e-8)

# Plot results
plt.figure(figsize=(10, 10))
plt.plot(0, 0, 'yo', label='Sun')
plt.plot(sol_earth.y[0], sol_earth.y[1], color='blue', linestyle='-', label="Earth's Orbit")
plt.plot([r_mars_initial], [0], 'r.', label='Mars Start')  # Initial position marker
plt.plot(sol_transfer.y[0], sol_transfer.y[1], color='orange', linestyle='--', label='Transfer Ellipse')
plt.plot(sol_final.y[0], sol_final.y[1], color='green', linestyle='-', label='Mars New Orbit')
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Hohmann Transfer Simulation: Bringing Mars Closer to Earth')
plt.legend()
plt.grid(True)
plt.show()

# Animation setup
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(0, 0, 'yo', label='Sun')
ax.plot(sol_earth.y[0], sol_earth.y[1], color='blue', linestyle='-', label="Earth's Orbit")
ax.plot(sol_final.y[0], sol_final.y[1], color='green', linestyle='-', label='Mars New Orbit')
ax.axis('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Animated Hohmann Transfer: Bringing Mars Closer to Earth')
ax.legend()
ax.grid(True)

# Mars position line and point
mars_line, = ax.plot([], [], color='orange', linestyle='--', label='Transfer Path')
mars_point, = ax.plot([], [], 'ro', label='Mars Position')

def init():
    mars_line.set_data([], [])
    mars_point.set_data([], [])
    return mars_line, mars_point

def animate(i):
    mars_line.set_data(sol_transfer.y[0][:i], sol_transfer.y[1][:i])
    mars_point.set_data([sol_transfer.y[0][i-1]], [sol_transfer.y[1][i-1]])
    return mars_line, mars_point

# Create animation (500 frames for smooth, interval=20ms)
anim = FuncAnimation(fig, animate, init_func=init, frames=len(sol_transfer.y[0]), interval=20, blit=True)

# Save as GIF (optional, for sharing on GitHub)
anim.save('mars_transfer.gif', writer=PillowWriter(fps=30))

plt.show()  # Still show static if wanted

print("Simulation completed successfully.")
