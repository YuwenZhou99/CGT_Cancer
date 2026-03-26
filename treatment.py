import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv


def benefit_function(p, steepness=8, threshold=0.3):
    return 1.0 / (1.0 + np.exp(-steepness * (p - threshold)))


def producer_fitness(p, cost=0.15, group_size=20, therapy_strength=0.0, steepness=8, threshold=0.3):
    n = group_size
    local_fraction_for_producer = (1 + (n - 1) * p) / n
    benefit = benefit_function(local_fraction_for_producer, steepness, threshold)
    return (1 - therapy_strength) * benefit - cost


def nonproducer_fitness(p, group_size=20, therapy_strength=0.0, steepness=8, threshold=0.3):
    n = group_size
    local_fraction_for_nonproducer = ((n - 1) * p) / n
    benefit = benefit_function(local_fraction_for_nonproducer, steepness, threshold)
    return (1 - therapy_strength) * benefit


def no_treatment_policy(p, t, d_max=0.0, p_crit=0.5):
    return 0.0


def constant_treatment_policy(p, t, d_max=0.5, p_crit=0.5):
    return d_max


def threshold_treatment_policy(p, t, d_max=0.5, p_crit=0.5):
    return d_max if p > p_crit else 0.0


def simulate_controlled_dynamics(
    p0=0.5,
    steps=200,
    dt=0.2,
    cost=0.15,
    group_size=20,
    steepness=8,
    threshold=0.3,
    policy=None,
    d_max=0.5,
    p_crit=0.5,
    sigma=0.01,
    p_target=0.1
):
    if policy is None:
        policy = no_treatment_policy

    p = p0
    p_history = []
    d_history = []
    wp_history = []
    wd_history = []
    objective = 0.0
    reached_target = False
    stopping_time = steps

    for t in range(steps):
        current_therapy = policy(p, t, d_max=d_max, p_crit=p_crit)

        wp = producer_fitness(
            p, cost=cost, group_size=group_size,
            therapy_strength=current_therapy,
            steepness=steepness, threshold=threshold
        )
        wd = nonproducer_fitness(
            p, group_size=group_size,
            therapy_strength=current_therapy,
            steepness=steepness, threshold=threshold
        )

        dp = dt * p * (1 - p) * (wp - wd)
        p = np.clip(p + dp, 0.0, 1.0)

        p_history.append(p)
        d_history.append(current_therapy)
        wp_history.append(wp)
        wd_history.append(wd)

        objective += (current_therapy + sigma) * dt

        if p <= p_target:
            reached_target = True
            stopping_time = t + 1
            break

    if not reached_target:
        objective += 1000.0

    return {
        "producer_fraction": np.array(p_history),
        "nonproducer_fraction": 1 - np.array(p_history),
        "producer_fitness": np.array(wp_history),
        "nonproducer_fitness": np.array(wd_history),
        "therapy": np.array(d_history),
        "objective": objective,
        "reached_target": reached_target,
        "stopping_time": stopping_time
    }


def save_policy_trajectory_csv(result, save_dir, filename):
    path = os.path.join(save_dir, filename)
    n = len(result["producer_fraction"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_step",
            "producer_fraction",
            "nonproducer_fraction",
            "producer_fitness",
            "nonproducer_fitness",
            "therapy"
        ])
        for t in range(n):
            writer.writerow([
                t,
                result["producer_fraction"][t],
                result["nonproducer_fraction"][t],
                result["producer_fitness"][t],
                result["nonproducer_fitness"][t],
                result["therapy"][t]
            ])


def save_policy_summary_csv(results, save_dir, filename="policy_summary.csv"):
    path = os.path.join(save_dir, filename)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "final_producer_fraction",
            "objective",
            "reached_target",
            "stopping_time"
        ])

        for name, res in results.items():
            writer.writerow([
                name,
                res["producer_fraction"][-1] if len(res["producer_fraction"]) > 0 else "",
                res["objective"],
                res["reached_target"],
                res["stopping_time"]
            ])


def plot_policy_comparison(
    p0=0.6,
    steps=200,
    dt=0.2,
    cost=0.05,
    group_size=20,
    steepness=8,
    threshold=0.3,
    d_max=0.5,
    p_crit=0.5,
    sigma=0.01,
    p_target=0.1,
    save_dir=None
):
    results = {
        "No treatment": simulate_controlled_dynamics(
            p0=p0, steps=steps, dt=dt, cost=cost,
            group_size=group_size, steepness=steepness, threshold=threshold,
            policy=no_treatment_policy, d_max=d_max, p_crit=p_crit,
            sigma=sigma, p_target=p_target
        ),
        "Constant treatment": simulate_controlled_dynamics(
            p0=p0, steps=steps, dt=dt, cost=cost,
            group_size=group_size, steepness=steepness, threshold=threshold,
            policy=constant_treatment_policy, d_max=d_max, p_crit=p_crit,
            sigma=sigma, p_target=p_target
        ),
        "Adaptive treatment": simulate_controlled_dynamics(
            p0=p0, steps=steps, dt=dt, cost=cost,
            group_size=group_size, steepness=steepness, threshold=threshold,
            policy=threshold_treatment_policy, d_max=d_max, p_crit=p_crit,
            sigma=sigma, p_target=p_target
        )
    }

    if save_dir is not None:
        save_policy_trajectory_csv(results["No treatment"], save_dir, "trajectory_no_treatment.csv")
        save_policy_trajectory_csv(results["Constant treatment"], save_dir, "trajectory_constant_treatment.csv")
        save_policy_trajectory_csv(results["Adaptive treatment"], save_dir, "trajectory_adaptive_treatment.csv")
        save_policy_summary_csv(results, save_dir)

    plt.figure(figsize=(10, 5))
    for name, res in results.items():
        plt.plot(res["producer_fraction"], label=name, linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("Producer fraction")
    plt.title("Producer dynamics under different treatment policies")
    plt.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "policy_comparison_dynamics.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    for name, res in results.items():
        plt.plot(res["therapy"], label=name, linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("Treatment intensity")
    plt.title("Treatment schedule under different policies")
    plt.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "policy_comparison_treatment.png"), dpi=300, bbox_inches="tight")
    plt.show()

    names = list(results.keys())
    values = [results[name]["objective"] for name in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Objective value")
    plt.title("Policy comparison by objective")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "policy_comparison_objective.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return results


def adaptive_parameter_sweep_figure(
    p0=0.6,
    steps=400,
    dt=0.2,
    cost=0.05,
    group_size=20,
    steepness=8,
    threshold=0.3,
    d_values=None,
    pcrit_values=None,
    sigma=0.01,
    p_target=0.45,
    save_dir=None
):
    if d_values is None:
        d_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    if pcrit_values is None:
        pcrit_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

    objective_grid = np.zeros((len(pcrit_values), len(d_values)))
    final_p_grid = np.zeros((len(pcrit_values), len(d_values)))

    rows = []

    for i, p_crit in enumerate(pcrit_values):
        for j, d_max in enumerate(d_values):
            res = simulate_controlled_dynamics(
                p0=p0,
                steps=steps,
                dt=dt,
                cost=cost,
                group_size=group_size,
                steepness=steepness,
                threshold=threshold,
                policy=threshold_treatment_policy,
                d_max=d_max,
                p_crit=p_crit,
                sigma=sigma,
                p_target=p_target
            )

            final_p = res["producer_fraction"][-1] if len(res["producer_fraction"]) > 0 else np.nan

            objective_grid[i, j] = res["objective"]
            final_p_grid[i, j] = final_p

            rows.append([
                d_max,
                p_crit,
                final_p,
                res["objective"],
                res["reached_target"],
                res["stopping_time"]
            ])

    if save_dir is not None:
        csv_path = os.path.join(save_dir, "adaptive_parameter_sweep.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "d_max",
                "p_crit",
                "final_producer_fraction",
                "objective",
                "reached_target",
                "stopping_time"
            ])
            writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    grids = [
        (final_p_grid, "Final producer fraction", "{:.3f}"),
        (objective_grid, "Objective value", "{:.1f}")
    ]

    for ax, (grid, title, fmt) in zip(axes.flat, grids):
        im = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_xticks(range(len(d_values)))
        ax.set_xticklabels([str(x) for x in d_values])
        ax.set_yticks(range(len(pcrit_values)))
        ax.set_yticklabels([str(x) for x in pcrit_values])
        ax.set_xlabel(r"$d_{\max}$")
        ax.set_ylabel(r"$p_{\mathrm{crit}}$")
        ax.set_title(title)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                ax.text(
                    j, i, fmt.format(value),
                    ha="center", va="center",
                    color="white" if value < (np.nanmin(grid) + np.nanmax(grid)) / 2 else "black",
                    fontsize=9
                )

        fig.colorbar(im, ax=ax)

    plt.suptitle("Adaptive therapy parameter sweep", fontsize=16)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "adaptive_parameter_sweep_2panel.png"),
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    return {
        "objective_grid": objective_grid,
        "final_p_grid": final_p_grid,
        "d_values": d_values,
        "pcrit_values": pcrit_values
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treatment extension for producer/non-producer cancer model")
    parser.add_argument("--p0", type=float, default=0.6, help="Initial producer proportion")
    parser.add_argument("--steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.2, help="Evolution step size")
    parser.add_argument("--cost", type=float, default=0.05, help="Production cost")
    parser.add_argument("--group_size", type=int, default=20, help="Group size")
    parser.add_argument("--steepness", type=float, default=8, help="Sigmoid steepness")
    parser.add_argument("--threshold", type=float, default=0.3, help="Sigmoid threshold")
    parser.add_argument("--d_max", type=float, default=0.5, help="Maximum treatment intensity")
    parser.add_argument("--p_crit", type=float, default=0.45, help="Adaptive treatment threshold")
    parser.add_argument("--sigma", type=float, default=0.01, help="Time penalty in the objective")
    parser.add_argument("--p_target", type=float, default=0.45, help="Recovery target for producer fraction")
    parser.add_argument("--output_dir", type=str, default="plots_treatment", help="Folder to save plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    plot_policy_comparison(
        p0=args.p0,
        steps=args.steps,
        dt=args.dt,
        cost=args.cost,
        group_size=args.group_size,
        steepness=args.steepness,
        threshold=args.threshold,
        d_max=args.d_max,
        p_crit=args.p_crit,
        sigma=args.sigma,
        p_target=args.p_target,
        save_dir=args.output_dir
    )

    adaptive_parameter_sweep_figure(
        p0=args.p0,
        steps=args.steps,
        dt=args.dt,
        cost=args.cost,
        group_size=args.group_size,
        steepness=args.steepness,
        threshold=args.threshold,
        d_values=[0.2, 0.3, 0.4, 0.5, 0.6],
        pcrit_values=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
        sigma=args.sigma,
        p_target=args.p_target,
        save_dir=args.output_dir
    )