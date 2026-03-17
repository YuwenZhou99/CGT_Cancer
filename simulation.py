import argparse
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


def simulate_dynamics(p0=0.5, steps=200, dt=0.2, cost=0.15, group_size = 10, therapy_strength=0.0, therapy_start=None, steepness=10, threshold=0.3):
    """
    Replicator dynamics for producer fraction p
    """
    p = p0 # proportion of producers
    p_history=[]
    d_history=[]
    wp_history=[]
    wd_history=[]
    therapy_history=[]

    for t in range(steps):
        current_therapy=0.0
        if therapy_start is not None and t >= therapy_start:
            current_therapy = therapy_strength

        # producer fitness
        wp = producer_fitness(p, cost=cost, group_size=group_size, therapy_strength=current_therapy, steepness=steepness, threshold=threshold)
        # non-producer fitness
        wd = nonproducer_fitness(p, group_size=group_size, therapy_strength=current_therapy, steepness=steepness, threshold=threshold)

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


def plot_simulation(result, title="Cancer Simulation"):
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
    plt.title(title + "- Fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(producer_fitness_vals, label="Producer fitness")
    plt.plot(nonproducer_fitness_vals, label="Non-producer fitness")
    plt.plot(therapy_vals, "--", label="Therapy strength")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(title + "- Fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variables control in cancer simulation")
    parser.add_argument("--p0", type=float, default=0.05, help="Initial producer proportion")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.2, help="Evolution step size")
    parser.add_argument("--cost", type=float, default=0.05, help="Produce cost")
    parser.add_argument("--group_size", type=int, default=10, help="Group size")
    parser.add_argument("--therapy_strength", type=float, default=0.0, help="Therapy strength")
    parser.add_argument("--therapy_start", type=int, default=None, help="Therapy state step")

    args = parser.parse_args()

    result = simulate_dynamics(
        p0=args.p0,
        steps=args.steps,
        dt=args.dt,
        cost=args.cost,
        group_size=args.group_size,
        therapy_strength=args.therapy_strength,
        therapy_start=args.therapy_start
    )
    plot_simulation(result, title="Cancer Simulation")
