import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def metropolis_acceptance_prob(dy, temp):
    if dy <= 0:
        return np.float64(1.0)
    else:
        #print(np.exp(-dy/temp))
        return np.exp(-dy/temp)
    
def generate_exponential_annealing_schedule(gamma, t_init):
    return lambda k: np.power(gamma, k) * t_init

def generate_logarithmic_annealing_schedule(t_init):
    return lambda k: (t_init * np.log(2))/np.log(k+1) if k > 1 else t_init

# implements simulated annealing to minimize the function f
def simulated_annealing(f, x, T, t, k_max, avg_interval = -1, plot_interval=-1):

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ys = []
    acc_probs = []
    temps = []

    y_avgs = []
    y_sum = 0
    acc_prob_avgs = []
    acc_prob_sum = 0
    avg_xs = []

    x_current = x
    y_current = f(x_current)
    x_best, y_best = x_current, y_current
    for k in range(k_max):
        x_new = T(x_current)
        y_new = f(x_new)
        dy = y_new - y_current
        temp = t(k)
        acceptance_prob = metropolis_acceptance_prob(dy, temp)
        #todo: add way to seed RNG
        if dy <= 0 or np.random.random() < acceptance_prob:
            x_current, y_current = x_new, y_new
            #print("Took swap")
        if y_new < y_best:
            x_best, y_best = x_new, y_new


        # plotting
        ys.append(y_current)
        acc_probs.append(acceptance_prob)
        temps.append(temp)
        if k>0 and plot_interval>-1:
            if avg_interval > -1:
                # print(k)
                # print(y_current)
                # print(y_sum)
                y_sum += y_current
                acc_prob_sum += acceptance_prob
                if k%avg_interval == 0:
                    y_avgs.append(y_sum/avg_interval)
                    # print(y_sum)
                    # print(avg_interval)
                    # print(y_sum/avg_interval)
                    # print(y_avgs[-1])
                    # print(type(y_avgs[-1]))
                    acc_prob_avgs.append(acc_prob_sum/avg_interval)
                    avg_xs = [((i+1)*avg_interval)-(avg_interval//2) for i in range(k//avg_interval)]
                    y_sum, acc_prob_sum = 0, 0
            if k%plot_interval == 0:
                display.clear_output(wait = True)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                print(f"k = {k}")
                print(f"y_best: {y_best}")
                print(f"y_current: {y_current}")
                ax1.plot(ys)
                ax2.scatter(range(k+1), acc_probs, alpha = 0.5)
                ax3.plot(temps[1:])

                if avg_interval > -1:
                    # print(avg_xs)
                    # print(y_avgs)
                    ax1.plot(avg_xs, y_avgs, color='red')
                    ax2.plot(avg_xs, acc_prob_avgs, color='red')

                display.display(fig)
            
                    
    return x_best

# implements simulated annealing to minimize the function f
def simulated_annealing_with_greedy_search(f, x, T, t, k_max, avg_interval = -1, plot_interval=-1):

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ys = []
    acc_probs = []
    temps = []

    y_avgs = []
    y_sum = 0
    acc_prob_avgs = []
    acc_prob_sum = 0
    avg_xs = []

    x_current = x
    y_current = f(x_current)
    x_best, y_best = x_current, y_current
    for k in range(k_max):
        x_new = T(x_current)
        y_new = f(x_new)
        dy = y_new - y_current
        temp = t(k)
        acceptance_prob = metropolis_acceptance_prob(dy, temp)
        #todo: add way to seed RNG
        if dy <= 0 or np.random.random() < acceptance_prob:
            x_current, y_current = x_new, y_new
            #print("Took swap")
        if dy <= 0:
            i = 0
            while True:
                possible_swaps = x_current.possible_swaps()
                possible_sels = [x_current.copy() for _ in possible_swaps]
                for sel, swap in zip(possible_sels, possible_swaps):
                    sel.swap_players(swap)
                best_sel = possible_sels[np.argmin([f(sel) for sel in possible_sels])]
                if f(best_sel) > f(x_current): break
                #print(f(x_current))
                x_current, y_current = best_sel, f(best_sel)
                i += 1
            #print(f"number of iterations for greedy search: {i}")
        if y_current < y_best:
            x_best, y_best = x_current, y_current 



        # plotting
        ys.append(y_current)
        acc_probs.append(acceptance_prob)
        temps.append(temp)
        if k>0 and plot_interval>-1:
            if avg_interval > -1:
                # print(k)
                # print(y_current)
                # print(y_sum)
                y_sum += y_current
                acc_prob_sum += acceptance_prob
                if k%avg_interval == 0:
                    y_avgs.append(y_sum/avg_interval)
                    # print(y_sum)
                    # print(avg_interval)
                    # print(y_sum/avg_interval)
                    # print(y_avgs[-1])
                    # print(type(y_avgs[-1]))
                    acc_prob_avgs.append(acc_prob_sum/avg_interval)
                    avg_xs = [((i+1)*avg_interval)-(avg_interval//2) for i in range(k//avg_interval)]
                    y_sum, acc_prob_sum = 0, 0
            if k%plot_interval == 0:
                display.clear_output(wait = True)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                print(f"k = {k}")
                print(f"y_best: {y_best}")
                print(f"y_current: {y_current}")
                ax1.plot(ys)
                ax2.scatter(range(k+1), acc_probs, alpha = 0.5)
                ax3.plot(temps[1:])

                if avg_interval > -1:
                    # print(avg_xs)
                    # print(y_avgs)
                    ax1.plot(avg_xs, y_avgs, color='red')
                    ax2.plot(avg_xs, acc_prob_avgs, color='red')

                display.display(fig)
            
                    
    return x_best

# implements simulated annealing to minimize the function f
def simulated_annealing_with_tabu(f, x, T, t, k_max, tabu_len, avg_interval = -1, plot_interval=-1):

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    tabu_list = []

    ys = []
    acc_probs = []
    temps = []

    y_avgs = []
    y_sum = 0
    acc_prob_avgs = []
    acc_prob_sum = 0
    avg_xs = []

    x_current = x
    y_current = f(x_current)
    x_best, y_best = x_current, y_current
    for k in range(k_max):
        x_new, move = T(x_current, tabu_list)
        y_new = f(x_new)
        dy = y_new - y_current
        temp = t(k)
        acceptance_prob = metropolis_acceptance_prob(dy, temp)
        #todo: add way to seed RNG
        if dy <= 0 or np.random.random() < acceptance_prob:
            x_current, y_current = x_new, y_new
            tabu_list.append(move)
            if len(tabu_list) > tabu_len:
                tabu_list.pop(0)
            #print("Took swap")
        if y_new < y_best:
            x_best, y_best = x_new, y_new


        # plotting
        ys.append(y_current)
        acc_probs.append(acceptance_prob)
        temps.append(temp)
        if k>0 and plot_interval>-1:
            if avg_interval > -1:
                # print(k)
                # print(y_current)
                # print(y_sum)
                y_sum += y_current
                acc_prob_sum += acceptance_prob
                if k%avg_interval == 0:
                    y_avgs.append(y_sum/avg_interval)
                    # print(y_sum)
                    # print(avg_interval)
                    # print(y_sum/avg_interval)
                    # print(y_avgs[-1])
                    # print(type(y_avgs[-1]))
                    acc_prob_avgs.append(acc_prob_sum/avg_interval)
                    avg_xs = [((i+1)*avg_interval)-(avg_interval//2) for i in range(k//avg_interval)]
                    y_sum, acc_prob_sum = 0, 0
            if k%plot_interval == 0:
                
                display.clear_output(wait = True)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                print(f"k = {k}")
                print(f"y_best: {y_best}")
                print(f"y_current: {y_current}")
                ax1.plot(ys)
                ax2.scatter(range(k+1), acc_probs, alpha = 0.5)
                ax3.plot(temps[1:])

                if avg_interval > -1:
                    # print(avg_xs)
                    # print(y_avgs)
                    ax1.plot(avg_xs, y_avgs, color='red')
                    ax2.plot(avg_xs, acc_prob_avgs, color='red')

                display.display(fig)
            
                    
    return x_best


# implements simulated annealing to minimize the function f
def simulated_annealing_with_adaptive_cooling(f, x, T, t, t_init, window_size, k_max, avg_interval = -1, plot_interval=-1):

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ys = []
    acc_probs = []
    temps = []

    y_avgs = []
    y_sum = 0
    acc_prob_avgs = []
    acc_prob_sum = 0
    avg_xs = []

    x_current = x
    y_current = f(x_current)
    x_best, y_best = x_current, y_current
    acceptances = []
    temp = t_init
    for k in range(k_max):
        x_new = T(x_current)
        y_new = f(x_new)
        dy = y_new - y_current

        acceptance_rate = 0.5 if len(acceptances) < window_size else np.sum(acceptances)/len(acceptances)
        print(acceptance_rate)
        #print(len(acceptances))
        temp = t(temp, acceptance_rate)
        acceptance_prob = metropolis_acceptance_prob(dy, temp)
        #todo: add way to seed RNG
        if dy <= 0 or np.random.random() < acceptance_prob:
            x_current, y_current = x_new, y_new
            acceptances.append(1)
            #print("Took swap")
        else:
            acceptances.append(0)
        if len(acceptances) > window_size:
            acceptances.pop(0)
        if y_new < y_best:
            x_best, y_best = x_new, y_new


        # plotting
        ys.append(y_current)
        acc_probs.append(acceptance_prob)
        temps.append(temp)
        if k>0 and plot_interval>-1:
            if avg_interval > -1:
                # print(k)
                # print(y_current)
                # print(y_sum)
                y_sum += y_current
                acc_prob_sum += acceptance_prob
                if k%avg_interval == 0:
                    y_avgs.append(y_sum/avg_interval)
                    # print(y_sum)
                    # print(avg_interval)
                    # print(y_sum/avg_interval)
                    # print(y_avgs[-1])
                    # print(type(y_avgs[-1]))
                    acc_prob_avgs.append(acc_prob_sum/avg_interval)
                    avg_xs = [((i+1)*avg_interval)-(avg_interval//2) for i in range(k//avg_interval)]
                    y_sum, acc_prob_sum = 0, 0
            if k%plot_interval == 0:
                display.clear_output(wait = True)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                print(f"k = {k}")
                print(f"y_best: {y_best}")
                print(f"y_current: {y_current}")
                ax1.plot(ys)
                ax2.scatter(range(k+1), acc_probs, alpha = 0.5)
                ax3.plot(temps[1:])

                if avg_interval > -1:
                    # print(avg_xs)
                    # print(y_avgs)
                    ax1.plot(avg_xs, y_avgs, color='red')
                    ax2.plot(avg_xs, acc_prob_avgs, color='red')

                display.display(fig)
            
                    
    return x_best