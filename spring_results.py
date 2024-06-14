import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pickle

def plot_compliance_vs_displacements(results):
    num_plots = 4
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(num_plots, 2, width_ratios=[1, 3], wspace=0.1, hspace=0.1)

    for i in range(num_plots):
        ax = fig.add_subplot(gs[i, 0])
        if i == 0:
            dx, solution = results[0]
            mesh = solution.mesh
            rho = solution.get_values('rho')
            mesh.plot_colored(rho, fig=fig, ax=ax, title="Original", cbar_label="rho")
        else:
            result = results[(2*i-1) % len(results)]
            dx, solution = result
            compliance = solution.get_values('total_compliance')
            solution.plot_deformed('rho', ax=ax, title=f'Disp.={dx}, E={compliance:.4f}')

    ax_large = fig.add_subplot(gs[:, 1])
    x_values = [result[0] for result in results]
    y_values = [result[1].get_values('total_compliance') for result in results]
    plt.plot(x_values, y_values)
    ax_large.set_title('Displacement vs. Energy')
    ax_large.set_xlabel('Displacement')
    ax_large.set_ylabel('Compliance')

    plt.tight_layout()
    plt.savefig("temp4.png")
    plt.close()

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

for dx, soln in results:
    force = soln.get_values('force_vertices')
    