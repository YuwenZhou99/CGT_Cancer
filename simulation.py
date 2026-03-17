import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def benefit_function(p, steepness=10, threshold=0.3):
    """
    Sigmoid public-good benefit
    """
    return 1.0 / (1.0 + np.exp(-steepness * (p - threshold)))


def producer_fitness(p, cost=0.15, group_size=10, therapy_strength=0.0, steepness=10, threshold=0.3):
    """
    A producer is in a local group of size n.
    Among the other n-1 cells, expected producer fraction is p.
    Since this focal cell is itself a producer, its local producer count is 1 + (n-1)*p
    """
    n = group_size
    local_fraction_for_producer = (1 + (n - 1) * p) / n
    benefit = benefit_function(local_fraction_for_producer, steepness, threshold)
    effective_benefit = (1 - therapy_strength) * benefit
    return effective_benefit - cost


def nonproducer_fitness(p, group_size=10, therapy_strength=0.0, steepness=10, threshold=0.3):
    """
    A non-producer does not contribute itself.
    Among the other n-1 cells, expected producer fraction is p.
    So its local producer count is (n-1)*p
    """
    n = group_size
    local_fraction_for_nonproducer = ((n - 1) * p) / n
    benefit = benefit_function(local_fraction_for_nonproducer, steepness, threshold)
    effective_benefit = (1 - therapy_strength) * benefit
    return effective_benefit


def simulate_dynamics(p0=0.5, steps=200, dt=0.2, cost=0.15, group_size=10,
                      therapy_strength=0.0, therapy_start=None, steepness=10, threshold=0.3):
    """
    Replicator dynamics for producer fraction p
    """
    p = p0  # proportion of producers
    p_history = []
    d_history = []
    wp_history = []
    wd_history = []
    therapy_history = []

    for t in range(steps):
        current_therapy = 0.0
        if therapy_start is not None and t >= therapy_start:
            current_therapy = therapy_strength

        # producer fitness
        wp = producer_fitness(
            p, cost=cost, group_size=group_size,
            therapy_strength=current_therapy,
            steepness=steepness, threshold=threshold
        )

        # non-producer fitness
        wd = nonproducer_fitness(
            p, group_size=group_size,
            therapy_strength=current_therapy,
            steepness=steepness, threshold=threshold
        )

        # the change ratio of producers in this round
        dp = dt * p * (1 - p) * (wp - wd)
        p = np.clip(p + dp, 0.0, 1.0)

        p_history.append(p)
        d_history.append(1 - p)
        wp_history.append(wp)
        wd_history.append(wd)
        therapy_history.append(current_therapy)

    return {
        "producer_fraction": np.array(p_history),
        "nonproducer_fraction": np.array(d_history),
        "producer_fitness": np.array(wp_history),
        "nonproducer_fitness": np.array(wd_history),
        "therapy": np.array(therapy_history)
    }


def find_equilibria(cost=0.05, group_size=10, therapy_strength=0.0,
                    steepness=10, threshold=0.3, num_points=1000):
    """
    Approximate interior equilibria where Wp(p) = Wd(p).
    We detect sign changes in Wp - Wd on a dense grid.
    """
    p_grid = np.linspace(0, 1, num_points)
    diff = []

    for p in p_grid:
        wp = producer_fitness(
            p, cost=cost, group_size=group_size,
            therapy_strength=therapy_strength,
            steepness=steepness, threshold=threshold
        )
        wd = nonproducer_fitness(
            p, group_size=group_size,
            therapy_strength=therapy_strength,
            steepness=steepness, threshold=threshold
        )
        diff.append(wp - wd)

    diff = np.array(diff)
    eq_points = []

    for i in range(len(p_grid) - 1):
        if abs(diff[i]) < 1e-8:
            eq_points.append(p_grid[i])
        elif diff[i] * diff[i + 1] < 0:
            # linear interpolation for approximate crossing
            p1, p2 = p_grid[i], p_grid[i + 1]
            d1, d2 = diff[i], diff[i + 1]
            p_star = p1 - d1 * (p2 - p1) / (d2 - d1)
            eq_points.append(p_star)

    return eq_points


def plot_simulation(result, title="Cancer Simulation", save_dir=None):
    producer_fraction = result["producer_fraction"]
    nonproducer_fraction = result["nonproducer_fraction"]
    producer_fitness_vals = result["producer_fitness"]
    nonproducer_fitness_vals = result["nonproducer_fitness"]
    therapy_vals = result["therapy"]

    plt.figure(figsize=(10, 5))
    plt.plot(producer_fraction, label="Producer fraction")
    plt.plot(nonproducer_fraction, label="Non-producer fraction")
    plt.xlabel("Time step")
    plt.ylabel("Fraction")
    plt.title(title + " - Fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "simulation_fraction.png"),
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(producer_fitness_vals, label="Producer fitness")
    plt.plot(nonproducer_fitness_vals, label="Non-producer fitness")
    plt.plot(therapy_vals, "--", label="Therapy strength")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(title + " - Fitness over time")
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "simulation_fitness.png"),
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()


def plot_fitness_vs_fraction(cost=0.05, group_size=10, therapy_strength=0.0,
                             steepness=10, threshold=0.3,
                             title="Fitness vs fraction of producers", 
                             save_dir=None):
    """
    Plot a paper-like figure:
    x-axis = fraction of producers
    y-axis = fitness
    """
    p_grid = np.linspace(0, 1, 500)

    wp_vals = []
    wd_vals = []

    for p in p_grid:
        wp = producer_fitness(
            p, cost=cost, group_size=group_size,
            therapy_strength=therapy_strength,
            steepness=steepness, threshold=threshold
        )
        wd = nonproducer_fitness(
            p, group_size=group_size,
            therapy_strength=therapy_strength,
            steepness=steepness, threshold=threshold
        )
        wp_vals.append(wp)
        wd_vals.append(wd)

    wp_vals = np.array(wp_vals)
    wd_vals = np.array(wd_vals)

    eq_points = find_equilibria(
        cost=cost, group_size=group_size,
        therapy_strength=therapy_strength,
        steepness=steepness, threshold=threshold
    )

    plt.figure(figsize=(8, 5))
    plt.plot(p_grid, wp_vals, label="Producer fitness", linewidth=2)
    plt.plot(p_grid, wd_vals, label="Non-producer fitness", linewidth=2, linestyle="--")
    plt.xlabel("Fraction of producers")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()

    # Mark approximate equilibrium points
    for p_star in eq_points:
        w_star = producer_fitness(
            p_star, cost=cost, group_size=group_size,
            therapy_strength=therapy_strength,
            steepness=steepness, threshold=threshold
        )
        plt.scatter([p_star], [w_star], s=40)
        plt.annotate(f"{p_star:.2f}", (p_star, w_star),
                     textcoords="offset points", xytext=(5, 5))

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "fitness_vs_fraction.png"),
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()


def plot_comparison(group_size=10, therapy_strength=0.0,
                               steepness=10, threshold=0.3,
                               high_cost=0.20, low_cost=0.05, 
                               save_dir=None):
    """
    Comparison Analysis
    """
    p_grid = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, cost, panel_title in zip(
        axes,
        [high_cost, low_cost],
        [f"High cost (c={high_cost})", f"Low cost (c={low_cost})"]
    ):
        wp_vals = []
        wd_vals = []

        for p in p_grid:
            wp = producer_fitness(
                p, cost=cost, group_size=group_size,
                therapy_strength=therapy_strength,
                steepness=steepness, threshold=threshold
            )
            wd = nonproducer_fitness(
                p, group_size=group_size,
                therapy_strength=therapy_strength,
                steepness=steepness, threshold=threshold
            )
            wp_vals.append(wp)
            wd_vals.append(wd)

        wp_vals = np.array(wp_vals)
        wd_vals = np.array(wd_vals)

        ax.plot(p_grid, wp_vals, label="Producer fitness", linewidth=2)
        ax.plot(p_grid, wd_vals, label="Non-producer fitness", linewidth=2, linestyle="--") 
        ax.set_xlabel("Fraction of producers")
        ax.set_title(panel_title)

        eq_points = find_equilibria(
            cost=cost, group_size=group_size,
            therapy_strength=therapy_strength,
            steepness=steepness, threshold=threshold
        )

        for p_star in eq_points:
            w_star = producer_fitness(
                p_star, cost=cost, group_size=group_size,
                therapy_strength=therapy_strength,
                steepness=steepness, threshold=threshold
            )
            ax.scatter([p_star], [w_star], s=35)

    axes[0].set_ylabel("Fitness")
    axes[0].legend()
    plt.suptitle("Fitness Comparison analysis")
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "fitness_comparison.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variables control in cancer simulation")
    parser.add_argument("--p0", type=float, default=0.05, help="Initial producer proportion")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.2, help="Evolution step size")
    parser.add_argument("--cost", type=float, default=0.05, help="Produce cost")
    parser.add_argument("--group_size", type=int, default=10, help="Group size")
    parser.add_argument("--therapy_strength", type=float, default=0.0, help="Therapy strength")
    parser.add_argument("--therapy_start", type=int, default=None, help="Therapy start step")
    parser.add_argument("--steepness", type=float, default=10, help="Sigmoid steepness")
    parser.add_argument("--threshold", type=float, default=0.3, help="Sigmoid threshold")
    parser.add_argument("--output_dir", type=str, default="plots", help="Folder to save plots")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Single fitness-vs-fraction plot
    plot_fitness_vs_fraction(
        cost=args.cost,
        group_size=args.group_size,
        therapy_strength=args.therapy_strength,
        steepness=args.steepness,
        threshold=args.threshold,
        title="Fitness vs fraction of producers", 
        save_dir=args.output_dir
    )

    # Comparison plots
    plot_comparison(
        group_size=args.group_size,
        therapy_strength=args.therapy_strength,
        steepness=args.steepness,
        threshold=args.threshold,
        high_cost=0.20,
        low_cost=0.05, 
        save_dir=args.output_dir
    )

    result = simulate_dynamics(
        p0=args.p0,
        steps=args.steps,
        dt=args.dt,
        cost=args.cost,
        group_size=args.group_size,
        therapy_strength=args.therapy_strength,
        therapy_start=args.therapy_start,
        steepness=args.steepness,
        threshold=args.threshold
    )

    plot_simulation(result, title="Cancer Simulation", save_dir=args.output_dir)
