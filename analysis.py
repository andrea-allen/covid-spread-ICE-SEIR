## Fill in here actually calling the model, plotting everything

from model.seir_model import solve_model, ModelParams
from model.plot_utils import plot_ts
from data.data_utils import find_min_mse, process_covid_data, plot_covid_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Helvetica'

def run_for_article(facility_name):
    # Parameter set dictionary for each facility we are looking at, so we can plug right into the model
    param_dict = {'ICE ADAMS COUNTY DETENTION CENTER': ModelParams(county_pop=30693, staff_pop=50, #a guess
                                                                        detention_pop=338, beta=0.4,sigma=0.5,
                                                                        gamma=1 / 10, gamma_ei=1 / 6.7,
                                                                        staff_work_shift=3, c_jail=3, c_0=3500,
                                                                        init_community_infections=150,
                                                                        init_detention_infections=2,arrest_rate=.0001,
                                                                        alos=.033),
                  'ICE KARNES COUNTY RESIDENTIAL CENTER': ModelParams(county_pop=15545, staff_pop=40,
                                                                        detention_pop=200, beta=0.4,sigma=0.5,
                                                                        gamma=1 / 10, gamma_ei=1 / 5,
                                                                        staff_work_shift=3, c_jail=2.0, c_0=2300,
                                                                        init_community_infections=200, #244 for feb 1, 233 for jan 1
                                                                        init_detention_infections=4,arrest_rate=.0006,
                                                                        alos=.033),
                  'ICE SOUTH TEXAS DETENTION COMPLEX': ModelParams(county_pop=20306, staff_pop=120,
                                                                        detention_pop=650, beta=0.4,sigma=0.6,
                                                                        gamma=1 / 10, gamma_ei=1 / 5.1,
                                                                        staff_work_shift=3, c_jail=1.7, c_0=5100,
                                                                        init_community_infections=233, #244 for feb 1, 233 for jan 1
                                                                        init_detention_infections=2,arrest_rate=.0003,
                                                                        alos=.033),
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':None, 'ICE WINN CORRECTIONAL CENTER': None}
    y_lim_dict = {'ICE ADAMS COUNTY DETENTION CENTER': [[10**(-3), 10**(-2), 10**(-1)], ['10', '100', '1,000'], 10**(-4), 10**(0)],
                  'ICE SOUTH TEXAS DETENTION COMPLEX': [[10**(-3), 10**(-2), 10**(-1)], ['.1%', '1%', '10%'], 10**(-3)-.0001, 10**(-1)+.1],
                  'ICE KARNES COUNTY RESIDENTIAL CENTER': [[10**(-3), 10**(-2), 10**(-1), 2**(-1)], ['.1%', '1%', '10%', '50%'], 10**(-3)-.0008, 10**(0)+1]}



    # facility_name = 'ICE ADAMS COUNTY DETENTION CENTER'
    #TODO update this to solve, and plot, separately
    model_soln = solve_model(model_params=param_dict[facility_name])
    active_cases_detainees, active_cases_county = process_covid_data(facility_name, '01-01-2021', '05-01-2021')


    ## Define which figure to plot on:
    # plt.figure('Staff cases', frameon=False)
    plt.figure('model', frameon=False)
    # plt.figure('raw data', frameon=False)
    plot_covid_data(active_cases_detainees, active_cases_county, y_lim_dict[facility_name])
    #TODO need to get these in somewhere:
    # N, model_params.county_pop, model_params.detention_pop, model_params.staff_pop
    plot_ts(model_soln, normalize=True, same_plot=True, show_staff=True, show_recovered=False, show_exposed=False,
            show_susceptible=False, num_days_plot=[0,120], show_county=True)
    plt.show()


def fit_model(active_cases_detainees, active_cases_county):
    mse_search_results = find_min_mse([active_cases_detainees, active_cases_county], c_jail_search=[3.7, 3.8, 3.9, 4], c_0_search=[2300, 2400, 2500, 2600, 2700, 2800],
                 init_community_infections_search=[250], init_detention_infections_search=[0], delay_range=[30,150])
    print('Results: ', mse_search_results[3], mse_search_results[4])
    print('Results county fit: ', mse_search_results[5], mse_search_results[6])


if __name__== '__main__':
    # facility_name = 'ICE SOUTH TEXAS DETENTION COMPLEX' #Do this for each one with rising cases
    facility_name = 'ICE KARNES COUNTY RESIDENTIAL CENTER' #Do this for each one with rising cases
    run_for_article(facility_name)