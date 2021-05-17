import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Helvetica'
import datetime
from model.seir_model import ModelParams, solve_model

def load_nyt_data():
    nyt_url_counties = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    df = pd.read_csv(nyt_url_counties, na_filter=True)
    return df


def load_ucla_data(fname=None):
    if fname is not None:
        data = pd.read_csv(fname)
        return data
    else:
        data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
        return data


def select_ice_facilities(data):
    ice_data = data[data['Jurisdiction'].str.contains('immigration', na=False)]
    print(ice_data.head(10))
    return ice_data

def process_covid_data(facility_name, start_date_str, end_date_str):
    ice_facilities = select_ice_facilities(load_ucla_data())
    covid_df = load_nyt_data()

    df = ice_facilities[ice_facilities['Name'].str.contains(f'{facility_name}', na=False)]

    facility_names = ['ICE ADAMS COUNTY DETENTION CENTER', 'ICE KARNES COUNTY RESIDENTIAL CENTER', 'ICE SOUTH TEXAS DETENTION COMPLEX',
                      'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER', 'ICE WINN CORRECTIONAL CENTER']
    county_dict = {'ICE ADAMS COUNTY DETENTION CENTER': 'Adams', 'ICE KARNES COUNTY RESIDENTIAL CENTER': 'Karnes', 'ICE SOUTH TEXAS DETENTION COMPLEX':'Frio',
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':'Frio', 'ICE WINN CORRECTIONAL CENTER':'Winn'}
    state_dict = {'ICE ADAMS COUNTY DETENTION CENTER' : 'Mississippi', 'ICE KARNES COUNTY RESIDENTIAL CENTER':'Texas',
                  'ICE SOUTH TEXAS DETENTION COMPLEX':'Texas', 'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':'Texas',
                  'ICE WINN CORRECTIONAL CENTER':'Louisiana'}
    pop_dict = {'ICE ADAMS COUNTY DETENTION CENTER': 30693, 'ICE KARNES COUNTY RESIDENTIAL CENTER': 15545, 'ICE SOUTH TEXAS DETENTION COMPLEX':20306,
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':20326, 'ICE WINN CORRECTIONAL CENTER': 14313}

    county = covid_df[covid_df['county'] == county_dict[facility_name]]
    county = county[county['state'] == state_dict[facility_name]]
    county_pop = pop_dict[facility_name]

    county_name = county.head(1)['county'].values[0]
    start_date = datetime.datetime.strptime('01-01-2021', '%m-%d-%Y')
    start_date = datetime.datetime.strptime(start_date_str, '%m-%d-%Y')
    start_date = pd.to_datetime(start_date)
    end_date = datetime.datetime.strptime('05-01-2021', '%m-%d-%Y')
    end_date = datetime.datetime.strptime(end_date_str, '%m-%d-%Y')
    end_date = pd.to_datetime(end_date)
    df['Date'] = pd.to_datetime(df['Date'])
    facility_mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
    facility = df.loc[facility_mask]

    county['date'] = pd.to_datetime(county['date'])
    county_mask = (county['date'] > start_date) & (county['date'] <= end_date)
    county = county.loc[county_mask]

    joined_df = county.join(facility.set_index('Date'), on='date', how='left')

    facility_pop = facility.head(1)['Population.Feb20'].values[0]
    if 'KARNES' in facility_name:
        facility_pop = 200
    active_cases_detainees = moving_average(np.array(joined_df['Residents.Active'].fillna(method='ffill').dropna())) / facility_pop

    active_cases_county = moving_average(np.array(joined_df['cases'].diff(10).fillna(method='bfill'))) / county_pop

    return active_cases_detainees, active_cases_county

def mean_squared_error(data, model):
    mse = np.square(np.subtract(data, model)).mean()
    return mse

def find_min_mse(data_list, c_jail_search=[3], c_0_search=[500], init_community_infections_search=[250],
                 init_detention_infections_search=[0], delay_range=[0,120], county_pop=20306, staff_pop=200, detention_pop=650):
    mse_ice_results = []
    mse_county_results = []
    mse_sum_results = []
    number_searches = len(c_jail_search) * len(c_0_search) * len(init_community_infections_search) * len(init_detention_infections_search)
    minimum_combo = 1000000
    min_county_fit = 1000000
    min_param_description = ''
    min_param_description_county = ''
    for c_jail in c_jail_search:
        for c_0 in c_0_search:
            for init_community_infections in init_community_infections_search:
                for init_detention_infections in init_detention_infections_search:
                    model_params = ModelParams(county_pop=county_pop, staff_pop=staff_pop,
                                      detention_pop=detention_pop, beta=0.4, sigma=0.5,
                                      gamma=1 / 10, gamma_ei=1 / 5.1,
                                      staff_work_shift=3, c_jail=c_jail, c_0=c_0,
                                      init_community_infections=init_community_infections,  # 244 for feb 1, 233 for jan 1
                                      init_detention_infections=init_detention_infections, arrest_rate=.0002,
                                      alos=.033)
                    model_soln = solve_model(model_params=model_params)
                    t_lim_min = min(np.where(model_soln[0] > delay_range[0])[0])
                    t_lim_max = max(np.where(model_soln[0] < delay_range[1])[0])
                    ice_data = data_list[0]
                    county_data = data_list[1]
                    model_ice = model_soln[3][t_lim_min:t_lim_min+len(ice_data)]
                    model_county = model_soln[7][t_lim_min:t_lim_min+len(county_data)]
                    mse_ice = mean_squared_error(ice_data, model_ice)
                    mse_c = mean_squared_error(county_data, model_county)
                    mse_ice_results.append(mse_ice)
                    mse_county_results.append(mse_c)
                    mse_sum_results.append(mse_ice + mse_c)
                    if mse_ice+mse_c < minimum_combo:
                        minimum_combo = mse_ice + mse_c
                        min_param_description = f'Min {minimum_combo} generated by c_jail {c_jail}, c_0 {c_0}, init_d {init_detention_infections}, init_c {init_community_infections}'
                        print(min_param_description)
                    if mse_c < min_county_fit:
                        min_county_fit = mse_c
                        min_param_description_county = f'Min {minimum_combo} generated by c_jail {c_jail}, c_0 {c_0}, init_d {init_detention_infections}, init_c {init_community_infections}'
    return mse_sum_results, mse_ice_results, mse_county_results, minimum_combo, min_param_description, min_county_fit, min_param_description_county

def moving_average(a, n=7):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_covid_data(active_cases_detainees, active_cases_county, y_lim_vals, show=False):
    base_color = '#555526'
    scatter_color = '#92926C'
    more_colors = [ "#D7790F", "#82CAA4", "#4C6788", "#84816F",
                "#71A9C9", "#AE91A8"]
    plt.rcParams['text.color'] = base_color

    plt.plot(np.arange(len(active_cases_detainees)), active_cases_detainees,
                label='Active Detainee Cases', lw=1, color=scatter_color)
    plt.scatter(np.arange(len(active_cases_county)),active_cases_county,
                label='Active County Cases', s=1, color=base_color)
    plt.legend(loc='upper left', frameon=False, )
    # plt.ylim([0, 0.15])
    plt.semilogy()
    plt.yticks(y_lim_vals[0], y_lim_vals[1])
    plt.ylim([y_lim_vals[2], y_lim_vals[3]])
    plt.xlabel(f'Days past January 1, 2021', color=base_color)
    # plt.xticks(np.arange(0, 80, 10), joined_df['date'][0, 10, 20, 30, 40, 50, 60, 70])
    # plt.ylim([.0005, .15])
    plt.box(on=None)
    plt.tick_params(axis='x', colors=base_color)
    plt.tick_params(axis='y', colors=base_color)

    if show:
        plt.show()