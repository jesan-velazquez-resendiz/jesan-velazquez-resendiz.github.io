---
title: 'Spin wave dispersion in a magnetic nanopillar with Ubermag'
collection: blog
category: micromagnetics
date: 2026-06-24
permalink: /blog/ubermag-nanopillar-spinwaves/
excerpt: 'Step-by-step micromagnetic simulation of spin wave propagation along a magnetic nanopillar using Ubermag (OOMMF). We reproduce the nmag probe example: relax the equilibrium state, excite spin waves with a short field pulse, and extract the dispersion relation via 2D FFT.'
tags:
  - micromagnetics
  - ubermag
  - spin waves
  - OOMMF
  - magnons
---

This tutorial shows how to simulate spin wave propagation along a magnetic nanopillar using
[Ubermag](https://ubermag.github.io/), a Python interface to OOMMF. We reproduce the spirit
of the classic [nmag nanopillar probe example](https://nmag.readthedocs.io/en/latest/example_nmagprobe/doc.html),
but implemented entirely in Python with Jupyter notebooks.

The goal is to excite spin waves (magnons) at one end of a nanowire with a short magnetic field
pulse, let them propagate, and recover the dispersion relation $$\omega(k)$$ via a 2D Fourier
transform in space and time.

The full notebooks are in the repository:
[`03a_nanopillar.ipynb`](https://github.com/jesan-velazquez-resendiz/jesan-velazquez-resendiz.github.io/blob/main/03a_nanopillar.ipynb) (simulation) and
[`03b_np_data.ipynb`](https://github.com/jesan-velazquez-resendiz/jesan-velazquez-resendiz.github.io/blob/main/03b_np_data.ipynb) (analysis).

---

## Background

### What is micromagnetism?

Micromagnetism describes magnetic materials at the mesoscale — below the scale of magnetic
domains but well above individual atoms. The fundamental quantity is the magnetization vector
field $$\mathbf{M}(\mathbf{r}, t)$$, constrained to have fixed magnitude $$|\mathbf{M}| = M_s$$
(the saturation magnetization) everywhere inside the material.

The time evolution is governed by the **Landau–Lifshitz–Gilbert (LLG) equation**:

$$\frac{d\mathbf{m}}{dt} = -\gamma_0 \, \mathbf{m} \times \mathbf{H}_\text{eff}
  + \alpha \, \mathbf{m} \times \frac{d\mathbf{m}}{dt}$$

where $$\mathbf{m} = \mathbf{M}/M_s$$ is the normalized magnetization, $$\gamma_0$$ is the
gyromagnetic ratio, $$\alpha$$ is the Gilbert damping constant, and $$\mathbf{H}_\text{eff}$$
is the effective field derived from the total energy:

$$\mathbf{H}_\text{eff} = -\frac{1}{\mu_0 M_s} \frac{\delta E}{\delta \mathbf{m}}$$

The total energy includes:
- **Exchange energy**: penalizes non-uniform magnetization, $$E_\text{ex} = A \int |\nabla \mathbf{m}|^2 \, dV$$
- **Demagnetization (dipolar) energy**: long-range magnetostatic interaction
- **Zeeman energy**: coupling to an external field, $$E_Z = -\mu_0 M_s \int \mathbf{m} \cdot \mathbf{H}_\text{ext} \, dV$$

### Spin waves

Small deviations from a uniform equilibrium magnetization are called **spin waves** or magnons.
In a thin magnetic wire aligned along $$x$$, transverse fluctuations $$\delta m_y$$ and
$$\delta m_z$$ propagate as plane waves $$\sim e^{i(kx - \omega t)}$$. The frequency–wavevector
relationship $$\omega(k)$$ is the **dispersion relation**, and measuring it reveals the exchange
constant, demagnetization effects, and the geometry of the sample.

The standard technique to measure $$\omega(k)$$ in simulation is to excite all wavevectors
simultaneously with a broadband pulse, record $$m_y(x, t)$$, and apply a 2D FFT.

---

## System setup

We use the following Ubermag packages:

```python
import discretisedfield as df   # geometry and fields
import micromagneticmodel as mm  # physics (energy, dynamics)
import oommfc as mc              # OOMMF computational engine

import numpy as np
import ipywidgets as widgets
from IPython.display import display
```

### Material parameters

The material is chosen to mimic a permalloy-like ferromagnet:

```python
Ms    = 0.86e6   # saturation magnetization (A/m)
A     = 13e-12   # exchange stiffness constant (J/m)
alpha = 0.5      # Gilbert damping (high value → fast relaxation)
```

The exchange length $$\ell_\text{ex} = \sqrt{2A / \mu_0 M_s^2} \approx 5.3 \, \text{nm}$$ sets
the characteristic length scale over which the magnetization can vary.

### Geometry: a cylindrical nanowire

The nanopillar is a cylinder of **radius 3 nm** and **length 600 nm**, aligned along $$x$$:

```python
region = df.Region(p1=(-300e-9, -3e-9, -3e-9), p2=(300e-9, 3e-9, 3e-9))
mesh   = df.Mesh(region=region, cell=(3e-9, 1e-9, 1e-9))
```

The bounding box is a rectangular cuboid, but the material only exists inside the cylinder.
We enforce this by making $$M_s$$ spatially dependent: it equals the real saturation
magnetization inside the cylinder and zero outside.

```python
def Ms_fun(point):
    """Return Ms inside the cylinder (y² + z² < r²), 0 outside."""
    x, y, z = point
    if (y**2 + z**2)**0.5 < 3e-9:
        return Ms
    else:
        return 0
```

The key argument `valid="norm"` tells Ubermag to treat cells where `Ms_fun` returns 0 as
vacuum — they are excluded from the physics entirely.

### Initial magnetization

We initialize the magnetization uniformly along $$x$$:

```python
sys_relax = mm.System(name='nanopillar_relax_05')
sys_relax.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms_fun, valid="norm")
```

`nvdim=3` means the field has 3 vector components ($$m_x, m_y, m_z$$).
`value=(1, 0, 0)` sets the initial direction; `norm=Ms_fun` rescales each cell
so $$|\mathbf{M}| = M_s$$ inside the cylinder and 0 outside.

---

## Phase 1: Finding the equilibrium state

Before studying dynamics we need the true ground-state magnetization. We include only
exchange and demagnetization energies (no external field), and use a minimization driver:

```python
sys_relax.energy   = mm.Exchange(A=A) + mm.Demag()
sys_relax.dynamics = mm.Damping(alpha=alpha) + mm.Precession(gamma0=mm.consts.gamma0)

md = mc.MinDriver()
md.drive(sys_relax, dirname='../../data/simulations/')
```

`MinDriver` solves the LLG equation with overdamped dynamics until the torque on every
cell falls below OOMMF's convergence threshold. The high damping value $$\alpha = 0.5$$
ensures fast convergence without oscillations. This run takes about **7 seconds** on a
standard laptop.

The result is an essentially uniform magnetization along $$x$$, slightly perturbed near
the ends by the demagnetization field (the so-called end domains).

---

## Phase 2: Spin wave dynamics

### Setting up the dynamic system

We create a new system initialized from the relaxed state. The damping is now set much
lower ($$\alpha = 0.05$$) so that spin waves can propagate without being damped out
before we can observe them:

```python
sys_dyn = mm.System(name='nanopillar_dynamics_05')

sys_dyn.dynamics = mm.Damping(alpha=0.05) + mm.Precession(gamma0=mm.consts.gamma0)

# copy the relaxed magnetization as the starting point
sys_dyn.m = df.Field(mesh, nvdim=3, value=sys_relax.m.array.copy(), norm=Ms, valid='norm')
```

### The excitation pulse

To excite spin waves across a broad range of wavevectors we apply a **spatially localized,
temporally short** magnetic field pulse. Localization in space → broad $$k$$-spectrum;
short duration → broad $$\omega$$-spectrum. This is the direct analogue of a delta-function
kick in time–space.

The pulse is applied only to the leftmost cell slice ($$x < x_\text{min} + \Delta x$$):

```python
pulse_boundary  = -300.0e-9 + mesh.cell[0]   # left end: x < -297 nm
pulse_amplitude = 1e5                          # 100 kA/m along y
pulse_duration  = 1e-12                        # 1 ps
total_time      = 200e-12                      # 200 ps total simulation
save_dt         = 0.5e-12                      # save snapshot every 0.5 ps

def H_pulse(point):
    """Spatially localized pulse: non-zero only at the left end."""
    x, y, z = point
    if x < pulse_boundary:
        return (0, pulse_amplitude, 0)   # along y
    else:
        return (0, 0, 0)

H_field = df.Field(mesh, nvdim=3, value=H_pulse)
```

With the pulse active, the energy is:

```python
sys_dyn.energy = mm.Exchange(A=A) + mm.Demag() + mm.Zeeman(H=H_field)
```

### Running in two stages

The simulation runs in **two stages** using `TimeDriver`:

```python
td = mc.TimeDriver()
```

**Stage 1 — pulse on (0 to 1 ps):**

```python
n_pulse = round(pulse_duration / save_dt)   # = 2 snapshots
td.drive(sys_dyn, t=pulse_duration, n=n_pulse,
         dirname='../../data/simulations/', verbose=2)
```

**Stage 2 — free evolution (1 ps to 200 ps):**

Once the pulse is off we remove the Zeeman term and let the spin waves propagate freely:

```python
sys_dyn.energy = mm.Exchange(A=A) + mm.Demag()   # no more Zeeman

remaining = total_time - pulse_duration           # 199 ps
n_rest    = round(remaining / save_dt)            # 398 snapshots

td.drive(sys_dyn, t=remaining, n=n_rest,
         dirname='../../data/simulations/', verbose=2)
```

Stage 1 takes ~7 s and stage 2 ~23 s. In total you end up with **400 magnetization
snapshots** covering 200 ps of dynamics, each storing the full 3-component vector field
on the mesh.

---

## Data analysis

The analysis lives in the second notebook. We use `micromagneticdata` to load the saved drives:

```python
import micromagneticdata as mdata

drive_eq    = mdata.Data(name="nanopillar_relax_05",    dirname='../../data/simulations/')[0]
drive_pulse = mdata.Data(name="nanopillar_dynamics_05", dirname='../../data/simulations/')[0]
drive_evo   = mdata.Data(name="nanopillar_dynamics_05", dirname='../../data/simulations/')[-1]
```

`[0]` selects the first drive (stage 1), `[-1]` the last (stage 2, free evolution).

### Extracting Δm_y(x, t)

Spin waves are transverse oscillations. We monitor $$m_y$$ along the centre of the
nanowire ($$y \approx 0$$, $$z \approx 0$$) and subtract the equilibrium value to isolate
the dynamic part:

```python
Ms      = 0.86e6
save_dt = 0.5e-12
t_max   = 100e-12        # analyse the first 100 ps
n_steps = int(t_max / save_dt)     # 200 points
t_pulse = 1e-12
n_pulse = int(t_pulse / save_dt)   # 2 points

arr0 = drive_evo[0].array           # shape (nx, ny, nz, 3)
nx, ny, nz, _ = arr0.shape

iy = ny // 2    # central y-index
iz = nz // 2    # central z-index

# equilibrium my along the wire
m0_y = drive_eq[0].array[:, iy, iz, 1] / Ms

# allocate 2D array: rows = time, columns = x position
dm_y = np.zeros((n_steps + n_pulse, nx))

# stage 1: pulse on
for i in range(n_pulse):
    m_y = drive_pulse[i].array[:, iy, iz, 1] / Ms
    dm_y[i] = m_y - m0_y

# stage 2: free evolution
for i in range(n_steps):
    m_y = drive_evo[i].array[:, iy, iz, 1] / Ms
    dm_y[i + n_pulse] = m_y - m0_y
```

Each `drive[i].array` has shape `(nx, ny, nz, 3)`. Index `1` selects $$m_y$$. We
normalize by $$M_s$$ to get dimensionless reduced magnetization.

### Space–time map

Plotting $$\Delta m_y(x, t)$$ as a 2D color map already shows spin wave packets
propagating away from the excitation point at the left end:

```python
times = np.arange(n_steps) * save_dt * 1e12                 # ps
x     = drive_evo.to_xarray().x.values * 1e9                 # nm

fig, ax = plt.subplots(figsize=(10, 5))
extent = [x[0], x[-1], times[0], times[-1]]
vmax = np.abs(dm_y).max()
im = ax.imshow(dm_y, aspect='auto', origin='lower', extent=extent,
               cmap='gnuplot', vmin=-vmax, vmax=vmax)
ax.set_xlabel('x (nm)')
ax.set_ylabel('t (ps)')
ax.set_title(r'$\Delta m_y(t, x)$ at $y \approx 0$, $z \approx 0$')
plt.colorbar(im, ax=ax, label=r'$\Delta m_y$')
plt.tight_layout()
plt.show()
```

You can see the spin wave front advancing from $$x = -300 \, \text{nm}$$ towards the
right end and reflecting back — the signature of a finite-length waveguide.

### Dispersion relation via 2D FFT

To extract $$\omega(k)$$ we Fourier-transform $$\Delta m_y(x, t)$$ in both dimensions.
Before doing so we apply a **Hanning window** in both time and space to suppress spectral
leakage at the boundaries:

```python
win_t = np.hanning(n_steps + n_pulse)[:, np.newaxis]   # shape (Nt, 1)
win_x = np.hanning(nx)[np.newaxis, :]                   # shape (1, Nx)
dm_y_w = dm_y * win_t * win_x
```

The 2D FFT and shift to centre the zero-frequency component:

```python
fft2d   = np.fft.fftshift(np.fft.fft2(dm_y_w[:, ::-1]))
fft_mag = np.abs(fft2d) / ((n_steps + n_pulse) * nx)
```

The `[::-1]` reversal along the spatial axis aligns the sign convention so that
positive $$k$$ corresponds to rightward-propagating waves.

Frequency axes:

```python
dt = save_dt
dx = drive_eq[0].mesh.cell[0] * 1e9       # cell size in nm

omega_GHz = np.fft.fftshift(np.fft.fftfreq(n_steps + n_pulse, d=dt)) * 1e-9   # GHz
k_per_nm  = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi            # rad/nm
```

Finally, plot the positive-frequency half of the spectrum:

```python
pos    = (omega_GHz >= 0) & (omega_GHz <= 500)
extent = [k_per_nm[0], k_per_nm[-1], omega_GHz[pos][0], omega_GHz[pos][-1]]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(fft_mag[pos, :], aspect='auto', origin='lower',
               extent=extent, cmap='gnuplot')
ax.set_xlabel(r'$k$ (rad/nm)')
ax.set_ylabel(r'$\omega$ (GHz)')
ax.set_title('Spin wave dispersion relation')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
```

The bright ridge in the $$(\omega, k)$$ plane is the spin wave dispersion relation. For
a thin wire dominated by exchange interactions it follows approximately:

$$\omega(k) \approx \gamma_0 \left( \mu_0 H_0 + \frac{2A}{\mu_0 M_s} k^2 \right)$$

The quadratic $$k^2$$ dependence is the hallmark of exchange-dominated spin waves, in
contrast to the linear dispersion of electromagnetic waves.

---

## Summary

| Step | Tool | Physical purpose |
|------|------|-----------------|
| Define geometry | `df.Region`, `df.Mesh` | Set up the 600 nm × 6 nm × 6 nm bounding box with 3×1×1 nm³ cells |
| Cylindrical mask | `Ms_fun` + `valid="norm"` | Restrict material to the cylinder of radius 3 nm |
| Relaxation | `mc.MinDriver` | Find the equilibrium magnetization (high $$\alpha = 0.5$$) |
| Pulse excitation | `mm.Zeeman` + `td.drive` (1 ps) | Inject spin waves broadband at the left end |
| Free evolution | `td.drive` (199 ps, $$\alpha = 0.05$$) | Let spin waves propagate along the wire |
| Space–time map | `plt.imshow` on $$\Delta m_y(x,t)$$ | Visualize propagation and reflection |
| Dispersion | 2D FFT + Hanning window | Extract $$\omega(k)$$ spectrum |

The same workflow generalizes to 2D films, disks, or any other geometry — simply change
the mesh and the `Ms_fun` mask. The 2D FFT approach is the standard tool for measuring
spin wave spectra in micromagnetic simulations and is directly comparable to
Brillouin light scattering (BLS) experiments.
