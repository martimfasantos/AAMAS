import subprocess
import matplotlib.pyplot as plt

# Create a directory to save the plots
output_dir = 'plots'
subprocess.run(['mkdir', '-p', output_dir])  # Create the directory

compare = True
mode = [0, 1, 2, 3]
times = [10, 100, 500]
fires = [3, 5]
n_agents = [2, 7]
seed = [False, True]
steps_incr = [None, 50]

for m in mode:
    for time in times:
        for f in fires:
            for n in n_agents:
                for s in seed:
                    for i in steps_incr:
                        commands = ['python', 'main.py', '--times',
                                    str(time), '--mode', str(m),
                                    '--fires', str(f),
                                    '--n_agents', str(n)]
                        if m == 0:
                            if time == 100: # Increase the number of steps
                                commands.extend(['--max_steps', str(1000)])
                            elif time == 500: # No need to run 500 times
                                continue
                        if compare:
                            commands.append('--compare')
                        if s:
                            commands.append('--seed')
                        if i and m != 0 and m != 1:
                            commands.extend(['--steps_incr', str(i)])
                        subprocess.run(commands)
