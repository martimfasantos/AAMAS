# Strategies for an Efficient Response to Wildfires

This repository contains the project developed in the Autonomous Agents and Multi-Agent Systems masters course @ IST.


Authors | Github
--------|--------
Martim Santos   | https://github.com/martimfasantos
Marina Gomes    | https://github.com/marinagomes02
Guilherme Gonçalves  | https://github.com/guilherme-goncalves793

**Project Grade:** 19/20

To acquire in-depth information regarding the project, please consult the [**report**](https://github.com/martimfasantos/AASMA/blob/main/AAMAS_Project_Report_2223.pdf).

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

You should create a virtual environment before installing the packages.  
Which can be done by running the following command:

```bash
python -m venv env
```

Then activate the virtual environment by running the following command:

```bash
source env/bin/activate
```

## Running the project

To run the project, run the following command (in the wildfires directory):

```bash
python main.py
```

Or you can run the project from the root directory by running the following command:

```bash
python wildfires/main.py
```

### Optional Arguments

- `--render`: Enable rendering of the game (*default: False*)
- `--debug`: Enable debug mode (*default: False*)
- `--times TIMES`: How many times to run the game (*default: 1*)
- `--max_steps MAX_STEPS`: How many steps in each episode (*default: 400*)
- `--fires FIRES`: How many fires in the environment (*default: 3*)
- `--steps_incr STEPS_INCR`: How many steps to increase the fire level by one (*default: None*)
- `--n_agents N_AGENTS`: How many agents to run with (*default: 2*)
- `--compare`: Plot graphs to compare teams (*default: False*)
- `--seed`: Use a seed for the environment (*default: False*)
- `--mode MODE`: Specify agent behavior mode:

**Note**: Please keep in mind that the parameters *FIRES* and *N_AGENTS* are conditioned by the size of the environment. Using excessively large values for these parameters may result in issues or unexpected behavior.
  
## Mode Options

| Modes                         |                         | Comparison Modes             |
|-------------------------------|-------------------------|------------------------------|
| 0: Randomly                   | 7: Social Conventions 1 | 0: Random vs Pseudo-random   |
| 1: Pseudo-randomly            | 8: Social Conventions 2 | 1: Greedy Heuristics         |
| 2: Greedy Heuristic 1         | 9: Social Conventions 3 | 2: Social Conventions        |
| 3: Greedy Heuristic 2         | 10: Role Based 1        | 3: Role Based                |
| 4: Greedy Heuristic 3         | 11: Role Based 2        | 4: All Better Agents         |
| 5: Greedy Heuristic 4         | 12: Role Based 3        | 5: Only Best Agents          |
| 6: Greedy Heuristic 5         | 13: Defined teams       |                              |

## Example

To run the game with default settings:

`python main.py`

To run the game with custom settings:

`python main.py --times 5 --max_steps 500 --fires 4 --n_agents 3 --compare --seed --mode 2 --render`

Feel free to customize the parameters according to your specific needs.

## License

This project is licensed under the [MIT License](LICENSE).

## Credits

This project is based on the following project

- [Level Based Foraging](https://github.com/semitable/lb-foraging)
