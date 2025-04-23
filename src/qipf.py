import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# KDE Wavefunction (Tensor)
def kde_wavefunction(t, A, sigma):
    t = t.unsqueeze(1)
    A = A.unsqueeze(0)
    ψ = torch.exp(-((t - A)**2) / (2 * sigma**2)).mean(dim=1)
    return ψ

# Hermite Polynomials (Tensor)
def hermite_polynomials(t, order):
    H = torch.zeros((len(t), order), device=device)
    H[:, 0] = 1
    if order > 1:
        H[:, 1] = 2 * t
    for n in range(2, order):
        H[:, n] = 2 * t * H[:, n-1] - 2 * (n-1) * H[:, n-2]
    return H

# Project onto Hermite Basis (Tensor)
def project_to_hermite_basis(ψ, H):
    return (ψ.unsqueeze(1) * H).T

# Laplacian computation (Tensor - explicit gradient twice)
def laplacian_in_data_space(ψ_k, dx):
    d2ψ_k = (ψ_k[:, :-2] - 2 * ψ_k[:, 1:-1] + ψ_k[:, 2:]) / dx**2
    d2ψ_k = F.pad(d2ψ_k, (1, 1), mode='replicate')  # explicitly match numpy padding
    return d2ψ_k

# Pipeline (Explicit Parameters, matches original exactly)
def qipf_pipeline(A, t_range=(-3,3), num_points=600, sigma=0.5, order=14):
    t = torch.linspace(t_range[0], t_range[1], num_points, device=device)
    dx = t[1] - t[0]

    ψ = kde_wavefunction(t, A, sigma)
    H = hermite_polynomials(t, order)
    ψ_k = project_to_hermite_basis(ψ, H)
    d2ψ_k = laplacian_in_data_space(ψ_k, dx)

    return t.cpu(), ψ.cpu(), ψ_k.cpu(), d2ψ_k.cpu()

# Visualization (same as original)
def plot_modes_and_laplacians(t, ψ, ψ_k, d2ψ_k, num_modes=6):
    """
    Plot ∇²ψ_k(t) for each Hermite mode, keeping relative amplitude across modes.
    
    Args:
        t (Tensor): 1D tensor of data-space grid
        ψ (Tensor): 1D KDE wavefunction
        ψ_k (Tensor): 2D tensor of Hermite mode projections (modes x len(t))
        d2ψ_k (Tensor): 2D tensor of Laplacians (modes x len(t))
        num_modes (int): Number of modes to visualize
    """
    colors = ['r', 'g', 'b', 'm', 'k', 'y']

    # Compute a shared normalization factor for Laplacian curves (as in numpy version)
    lap_max = torch.max(torch.tensor([torch.max(torch.abs(d2ψ_k[i])) for i in range(num_modes)]))

    plt.figure(figsize=(12, 6))

    # Plot each Laplacian mode normalized by the shared global max
    for i in range(num_modes):
        plt.plot(t, d2ψ_k[i] / lap_max, label=f"∇²ψ_{i+1}", color=colors[i % len(colors)])

    # Normalize ψ to [0, 1] for visual comparison
    ψ_norm = (ψ - ψ.min()) / (ψ.max() - ψ.min())
    plt.plot(t, ψ_norm, 'k--', linewidth=2, label="ψ(t) normalized")

    plt.title("Normalized ∇² Hermite Modes and ψ(t)")
    plt.xlabel("Data space (t)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main execution (Explicit parameters)
if __name__ == "__main__":
    T, f = 1000, 10
    x = torch.arange(0, T, device=device)
    A = torch.sin(2 * torch.pi * f * x / T)

    # Explicitly match original parameters
    t, ψ, ψ_k, d2ψ_k = qipf_pipeline(A, t_range=(-3,3), num_points=600, sigma=0.5, order=14)

    plot_modes_and_laplacians(t, ψ, ψ_k, d2ψ_k, num_modes=int(14 / 2) - 2)
