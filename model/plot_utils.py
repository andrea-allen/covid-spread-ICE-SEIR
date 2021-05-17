import matplotlib.pyplot as plt
import numpy as np

def plot_ts(time_series, N=400, county_pop=200, detention_pop=150, staff_pop=50, combine_staff=True,
            show_susceptible=False, show_recovered=False, community_separate_plot=False, normalize=False,
            same_plot=False, show_exposed=True, show_staff=True, show_county=True, num_days_plot=[0,100]):
    t_lim_min = min(np.where(time_series[0] > num_days_plot[0])[0])
    t_lim_max = max(np.where(time_series[0] < num_days_plot[1])[0])
    more_colors = [ "#D7790F", "#82CAA4", "#4C6788", "#84816F", "#71A9C9", "#AE91A8"]
    base_color = '#555526'
    scatter_color = '#92926C'
    plt.rcParams['text.color'] = base_color
    # Also would be good to have a way to interpolate this into incidence data to match reports
    time_series = np.array(time_series)
    # time_series = time_series.T
    if not same_plot:
        plt.figure('Detention Cases', frameon=False)
    plt.xlabel('Days')
    plt.ylabel('Percent of population')
    # Detention
    if not normalize:
        detention_pop = 1
    if show_susceptible:
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[1][t_lim_min:t_lim_max] * N / detention_pop, color='blue',
                 label='D - Susceptible')
    if show_exposed:
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[2][t_lim_min:t_lim_max] * N / detention_pop, color=more_colors[1], label='Model Detainees - Exposed')
    plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[3][t_lim_min:t_lim_max] * N / detention_pop, color=more_colors[0], label='Predicted Active Detainee Cases')
    if show_recovered:
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[4][t_lim_min:t_lim_max] * N / detention_pop, color='green',
                 label='D - Recovered')

    if not same_plot and show_staff==True:
        plt.figure('Staff cases', frameon=False)
    plt.xlabel('Days')
    plt.ylabel('Percent of population')
    if not combine_staff:
        # Community Staff
        if not normalize:
            staff_pop = 2
        if show_susceptible:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[9][t_lim_min:t_lim_max] * N / (staff_pop / 2), color='blue',
                     label='C Staff - Susceptible', ls='-.')
        if show_exposed:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[10][t_lim_min:t_lim_max] * N / (staff_pop / 2), color=more_colors[1],
                     label='C Staff - Exposed', ls='-.')
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[11][t_lim_min:t_lim_max] * N / (staff_pop / 2), color=more_colors[0],
                 label='C Staff - Infected', ls='-.')
        if show_recovered:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[12][t_lim_min:t_lim_max] * N / (staff_pop / 2), color='green',
                     label='C Staff - Recovered', ls='-.')

        # Detention Staff
        if show_susceptible:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[13][t_lim_min:t_lim_max] * N / (staff_pop / 2), color='blue',
                     label='D Staff - Susceptible', ls='--')
        if show_exposed:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[14][t_lim_min:t_lim_max] * N / (staff_pop / 2), color=more_colors[1],
                     label='D Staff - Exposed', ls='--')
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[15][t_lim_min:t_lim_max] * N / (staff_pop / 2), color=more_colors[0],
                 label='D Staff - Infected', ls='--')
        if show_recovered:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[16][t_lim_min:t_lim_max] * N / (staff_pop / 2), color='green',
                     label='D Staff - Recovered', ls='--')

    # # Staff combined (Comment out specific staff plots above)
    if combine_staff and show_staff:
        if not normalize:
            staff_pop = 1
        if show_susceptible:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], (time_series[9][t_lim_min:t_lim_max] + time_series[13][t_lim_min:t_lim_max]) * N / staff_pop,
                     color='blue', label='Staff - Susceptible', ls='-.')
        if show_exposed:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], (time_series[10][t_lim_min:t_lim_max] + time_series[14][t_lim_min:t_lim_max]) * N / staff_pop,
                     color=more_colors[1], label='Model Staff - Exposed', ls='-.')
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], (time_series[11][t_lim_min:t_lim_max] + time_series[15][t_lim_min:t_lim_max]) * N / staff_pop,
                 color=more_colors[0], label='Predicted Active Employee Cases', ls='-.')
        if show_recovered:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], (time_series[12][t_lim_min:t_lim_max] + time_series[16][t_lim_min:t_lim_max]) * N / staff_pop,
                     color='green', label='Staff - Recovered', ls='-.')

    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    plt.yticks([10**(-3), 10**(-2), 10**(-1)], ['10', '100', '1,000'])
    # plt.ylabel('Cases per 10,000 People', color=base_color)
    plt.ylabel('Percent of Population Actively Infected', color=base_color)
    # plt.legend(loc='upper left')
    plt.legend(loc='upper left')
    # cases per 10k instead?
    # plt.ylim([0, 1])
    # plt.tight_layout()

    if not same_plot:
        if community_separate_plot:
            plt.figure('community', frameon=False)
            plt.xlabel('Days')
            plt.ylabel('Percent of population')
    # Community
    if show_county:
        if not normalize:
            county_pop = 1
        if show_susceptible:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[5][t_lim_min:t_lim_max] * N / county_pop, color='blue', label='C - Susceptible',
                     ls=':')
        if show_exposed:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[6][t_lim_min:t_lim_max] * N / county_pop, color=more_colors[1], label='Model County - Exposed',
                     ls=':')
        plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[7][t_lim_min:t_lim_max] * N / county_pop, color=more_colors[0], label='Predicted Active County Cases', ls=':')
        if show_recovered:
            plt.plot(time_series[0][0:t_lim_max-t_lim_min], time_series[8][t_lim_min:t_lim_max] * N / county_pop, color='green', label='C - Recovered',
                     ls=':')

    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    # plt.legend(loc='upper left')
    plt.legend(loc='upper left', frameon=False)
    # plt.ylim([0, 1])
    # plt.tight_layout()
    # plt.show()