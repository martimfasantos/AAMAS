import math
import numpy as np
import matplotlib.pyplot as plt
from multi_agent_wildfires.agents import *



def generateTeams(mode, n_agents, compare=False):
    """
    Generates a dictionary of teams and their agents
    """

    if compare:
        if mode == 0: # Random vs PseudoRandom
            return {
                "Random Agents": {
                    "Helicopters": [RandomAgent for _ in range(n_agents // 2)],
                    "Firetrucks": [RandomAgent for _ in range(n_agents - n_agents // 2)]
                },
                "PseudoRandom Agents": {
                    "Helicopters": [PseudoRandomAgent for _ in range(n_agents // 2)],
                    "Firetrucks": [PseudoRandomAgent for _ in range(n_agents - n_agents // 2)]
                }
            }
        if mode == 1: # Greedy Comparison
            return {
                "Greedy H1 Agents": {
                    "Helicopters": [H1 for _ in range(n_agents // 2)],
                    "Firetrucks": [H1 for _ in range(n_agents - n_agents // 2)]
                },
                "Greedy H2 Agents": {
                    "Helicopters": [H2 for _ in range(n_agents // 2)],
                    "Firetrucks": [H2 for _ in range(n_agents - n_agents // 2)]
                },
                "Greedy H3 Agents": {
                    "Helicopters": [H3 for _ in range(n_agents // 2)],
                    "Firetrucks": [H3 for _ in range(n_agents - n_agents // 2)]
                },
                "Greedy H4 Agents": {
                    "Helicopters": [H4 for _ in range(n_agents // 2)],
                    "Firetrucks": [H4 for _ in range(n_agents - n_agents // 2)]
                },
                "Greedy H5 Agents": {
                    "Helicopters": [H5 for _ in range(n_agents // 2)],
                    "Firetrucks": [H5 for _ in range(n_agents - n_agents // 2)]
                }
            }
        if mode == 2: # Social Conventions Comparison
            return {
                "Convention C1 Agents": {
                    "Helicopters": [C1 for _ in range(n_agents // 2)],
                    "Firetrucks": [C1 for _ in range(n_agents - n_agents // 2)]
                },
                "Convention C2 Agents": {
                    "Helicopters": [C2 for _ in range(n_agents // 2)],
                    "Firetrucks": [C2 for _ in range(n_agents - n_agents // 2)]
                },
                "Convention C3 Agents": {
                    "Helicopters": [C3 for _ in range(n_agents // 2)],
                    "Firetrucks": [C3 for _ in range(n_agents - n_agents // 2)]
                },
                "Convention C4 Agents": {
                     "Helicopters": [C4 for _ in range(n_agents // 2)],
                    "Firetrucks": [C4 for _ in range(n_agents // 2)]
                }
            }
        if mode == 3: # Role Based Comparison
            return {
                "Role Based R1 Agents": {
                    "Helicopters": [R1 for _ in range(n_agents // 2)],
                    "Firetrucks": [R1 for _ in range(n_agents - n_agents // 2)]
                },
                "Role Based R2 Agents": {
                    "Helicopters": [R2 for _ in range(n_agents // 2)],
                    "Firetrucks": [R2 for _ in range(n_agents - n_agents // 2)]
                },
                "Role Based R3 Agents": {
                    "Helicopters": [R3 for _ in range(n_agents // 2)],
                    "Firetrucks": [R3 for _ in range(n_agents - n_agents // 2)]
                }
            }
        else:
            return {
                "PseudoRandom Agents": {
                    "Helicopters": [PseudoRandomAgent for _ in range(n_agents // 2)],
                    "Firetrucks": [PseudoRandomAgent for _ in range(n_agents - n_agents // 2)]
                },
                "Greedy H3 Agents": {
                    "Helicopters": [H3 for _ in range(n_agents // 2)],
                    "Firetrucks": [H3 for _ in range(n_agents - n_agents // 2)]
                },
                "Convention C3 Agents": {
                    "Helicopters": [C3 for _ in range(n_agents // 2)],
                    "Firetrucks": [C3 for _ in range(n_agents - n_agents // 2)]
                },
                "Role Based Agents": {
                    "Helicopters": [R1 for _ in range(n_agents // 2)],
                    "Firetrucks": [R1 for _ in range(n_agents - n_agents // 2)]
                }
            }
  
    else:
        if mode == 0:
            return {
                "Random Agents": {
                    "Helicopters": [RandomAgent for _ in range(n_agents // 2)],
                    "Firetrucks": [RandomAgent for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 1:
            return {
                "PseudoRandom Agents": {
                    "Helicopters": [PseudoRandomAgent for _ in range(n_agents // 2)],
                    "Firetrucks": [PseudoRandomAgent for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 2:
            return {
                "Greedy H1 Agents": {
                    "Helicopters": [H1 for _ in range(n_agents // 2)],
                    "Firetrucks": [H1 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 3:
            return {
                "Greedy H2 Agents": {
                    "Helicopters": [H2 for _ in range(n_agents // 2)],
                    "Firetrucks": [H2 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 4:
            return {
                "Greedy H3 Agents": {
                    "Helicopters": [H3 for _ in range(n_agents // 2)],
                    "Firetrucks": [H3 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 5:
            return {
                "Greedy H4 Agents": {
                    "Helicopters": [H4 for _ in range(n_agents // 2)],
                    "Firetrucks": [H4 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 6:
            return {
                "Greedy H5 Agents": {
                    "Helicopters": [H5 for _ in range(n_agents // 2)],
                    "Firetrucks": [H5 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 7:
            return {
                "Convention C1 Agents": {
                    "Helicopters": [C1 for _ in range(n_agents // 2)],
                    "Firetrucks": [C1 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 8:
            return {
                "Convention C2 Agents": {
                    "Helicopters": [C2 for _ in range(n_agents // 2)],
                    "Firetrucks": [C2 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 9:
            return {
                "Convention C3 Agents": {
                    "Helicopters": [C3 for _ in range(n_agents // 2)],
                    "Firetrucks": [C3 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 10:
            return {
                "Convention C4 Agents": {
                     "Helicopters": [C4 for _ in range(n_agents // 2)],
                    "Firetrucks": [C4 for _ in range(n_agents // 2)]
                }
            }
        elif mode == 11:
            return {
                "Role Based R1 Agents": {
                    "Helicopters": [R1 for _ in range(n_agents // 2)],
                    "Firetrucks": [R1 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 12:
            return {
                "Role Based R2 Agents": {
                    "Helicopters": [R2 for _ in range(n_agents // 2)],
                    "Firetrucks": [R2 for _ in range(n_agents - n_agents // 2)]
                }
            }
        elif mode == 13:
            return {
                "Role Based R3 Agents": {
                    "Helicopters": [R3 for _ in range(n_agents // 2)],
                    "Firetrucks": [R3 for _ in range(n_agents - n_agents // 2)]
                }
            }
        
        else:
            return {
                "PseudoRandom Agents": {
                    "Helicopters": [PseudoRandomAgent for _ in range(n_agents // 2)],
                    "Firetrucks": [PseudoRandomAgent for _ in range(n_agents - n_agents // 2)]
                },
                "Greedy H3 Agents": {
                    "Helicopters": [H3 for _ in range(n_agents // 2)],
                    "Firetrucks": [H3 for _ in range(n_agents - n_agents // 2)]
                },
                "Convention C3 Agents": {
                    "Helicopters": [C3 for _ in range(n_agents // 2)],
                    "Firetrucks": [C3 for _ in range(n_agents - n_agents // 2)]
                },
                "Role Based Agents": {
                    "Helicopters": [R1 for _ in range(n_agents // 2)],
                    "Firetrucks": [R1 for _ in range(n_agents - n_agents // 2)]
                }
            }


def z_table(confidence):
    """Hand-coded Z-Table

    Parameters
    ----------
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The z-value for the confidence level given.
    """
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]


def confidence_interval(mean, n, confidence):
    """Computes the confidence interval of a sample.

    Parameters
    ----------
    mean: float
        The mean of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The confidence interval.
    """
    return z_table(confidence) * (mean / math.sqrt(n))


def standard_error(std_dev, n, confidence):
    """Computes the standard error of a sample.

    Parameters
    ----------
    std_dev: float
        The standard deviation of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The standard error.
    """
    return z_table(confidence) * (std_dev / math.sqrt(n))


def plot_confidence_bar(names, means, std_devs, N, title, x_label, y_label, confidence, show=False, filename=None, colors=None, yscale=None):
    """Creates a bar plot for comparing different agents/teams.

    Parameters
    ----------

    names: Sequence[str]
        A sequence of names (representing either the agent names or the team names)
    means: Sequence[float]
        A sequence of means (one mean for each name)
    std_devs: Sequence[float]
        A sequence of standard deviations (one for each name)
    N: Sequence[int]
        A sequence of sample sizes (one for each name)
    title: str
        The title of the plot
    x_label: str
        The label for the x-axis (e.g. "Agents" or "Teams")
    y_label: str
        The label for the y-axis
    confidence: float
        The confidence level for the confidence interval
    show: bool
        Whether to show the plot
    filename: str
        If given, saves the plot to a file
    colors: Optional[Sequence[str]]
        A sequence of colors (one for each name)
    yscale: str
        The scale for the y-axis (default: linear)
    """

    errors = [standard_error(std_devs[i], N[i], confidence) for i in range(len(means))]
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, color=colors if colors is not None else "gray", ecolor='black', capsize=10)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.yaxis.grid(True)
    if yscale is not None:
        plt.yscale(yscale)
    plt.tight_layout()
    if filename is not None:
        fig.set_size_inches(2 * len(names), 5)
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


def compare_results(results, confidence=0.95, title="Agents Comparison", metric="Steps Per Episode", colors=None, filename=None):

    """Displays a bar plot comparing the performance of different agents/teams.

        Parameters
        ----------

        results: dict
            A dictionary where keys are the names and the values sequences of trials
        confidence: float
            The confidence level for the confidence interval
        title: str
            The title of the plot
        metric: str
            The name of the metric for comparison
        colors: Sequence[str]
            A sequence of colors (one for each agent/team)

        """

    names = list(results.keys())
    means = [result.mean() for result in results.values()]
    stds = [result.std() for result in results.values()]
    N = [result.size for result in results.values()]

    plot_confidence_bar(
        names=names,
        means=means,
        std_devs=stds,
        N=N,
        filename=filename,
        show= filename == None,
        title=title,
        x_label="", y_label=f"Avg. {metric}",
        confidence=confidence,
        colors=colors
    )



def compare_results_learning(results, confidence=0.95, title="Agents Comparison", metric="Steps Per Episode", colors=None):

    x = None

    for agent, agent_results in results.items():

        n_evaluations, n_eval_episodes = agent_results.shape

        x = tuple(range(n_evaluations))

        y = []
        yerr = []
        for evaluation in range(n_evaluations):
            result = agent_results[evaluation]
            mean = result.mean()
            std_dev = result.std()
            y.append(mean)
            yerr.append(standard_error(std_dev, n_eval_episodes, confidence))

        plt.errorbar(x, y, yerr=yerr, label=agent, ls="dotted", capsize=10, marker="o", color=colors[list(results.keys()).index(agent)] if colors is not None else None)

    x = np.array(x)
    plt.title(title)
    plt.legend()
    plt.xlabel("Evaluation Checkpoint")
    plt.ylabel(metric)
    plt.xticks(x, [int(x[i]) + 1 for i in range(x.size)])
    plt.grid()
    plt.show()
    plt.close()