import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np


class ModelParams:
    def __init__(self, beta, sigma, gamma, gamma_ei, staff_work_shift, c_jail, arrest_rate, alos,
                 county_pop, staff_pop, detention_pop, c_0, init_community_infections,
                 init_detention_infections):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.gamma_ei = gamma_ei
        self.gamma_asymptom = 1 / 6.7

        self.staff_work_shift = staff_work_shift  # 1/8 hrs = 1/(1/3)day = 3. Home shift = 1/16 hrs = 1/(2/3)day = 3/2
        self.staff_home_shift = 1 / ((24 - (1 / staff_work_shift) * 24) / 24)
        self.c_jail = c_jail
        self.c_0 = c_0
        self.arrest_rate = arrest_rate
        self.alos = alos

        self.county_pop = county_pop
        self.staff_pop = staff_pop
        self.detention_pop = detention_pop

        self.init_community_infections = init_community_infections
        self.init_detention_infections = init_detention_infections

    def callibrate(self):  # unclear whetehr to do staff pop yet
        beta_community_cal = self.c_0 * self.beta / (self.county_pop + self.staff_pop)
        beta_detention_cal = self.c_0 * self.c_jail * self.beta / (self.detention_pop + self.staff_pop)
        # beta_staff_cal = c_0 * self.beta / staff_pop
        self.beta_community = beta_community_cal
        self.beta_detention = beta_detention_cal
        # self.beta_staff = beta_staff_cal


class Model:
    def __init__(self, params, N, y_init):
        self.params = params
        self.N = N
        self.y_init = y_init
        self.time_series = []

    def solve_model(self):
        solution = scipy.integrate.solve_ivp(self.odes_seir_metapop, t_span=[0, 500], y0=self.y_init)
        self.time_series.append(solution.t)
        for i in range(len(self.y_init)):
            self.time_series.append(solution.y[i])
        return self.time_series

    def odes_seir_metapop(self, t, y):
        # y form: [s(t), e(t), i(t), r(t)]

        # Detention residents
        s_t_D = y[0]
        e_t_D = y[1]
        i_t_D = y[2]
        r_t_D = y[3]

        # Community
        s_t_C = y[4]
        e_t_C = y[5]
        i_t_C = y[6]
        r_t_C = y[7]

        # Employees in the commmunity
        s_t_O_C = y[8]
        e_t_O_C = y[9]
        i_t_O_C = y[10]
        r_t_O_C = y[11]

        # Employees in the facility
        s_t_O_D = y[12]
        e_t_O_D = y[13]
        i_t_O_D = y[14]
        r_t_O_D = y[15]

        if sum(y) > 1.01 or sum(y) < .99:
            print(sum(y))

        # detention
        ds_dt_D = - self.params.sigma * self.params.beta_detention * (s_t_D * e_t_D) - self.params.beta_detention * (
                s_t_D * i_t_D) \
                  - self.params.sigma * self.params.beta_detention * (s_t_D * e_t_O_D) - self.params.beta_detention * (
                          s_t_D * i_t_O_D) \
                  + self.params.arrest_rate * s_t_C
        de_dt_D = self.params.sigma * self.params.beta_detention * (s_t_D * e_t_D) + self.params.beta_detention * (
                s_t_D * i_t_D) - self.params.gamma_ei * e_t_D \
                  + self.params.sigma * self.params.beta_detention * (s_t_D * e_t_O_D) + self.params.beta_detention * (
                          s_t_D * i_t_O_D) + self.params.arrest_rate * e_t_C - self.params.gamma_asymptom * e_t_D
        di_dt_D = -self.params.gamma * i_t_D + self.params.gamma_ei * e_t_D \
                  - self.params.alos * i_t_D + self.params.arrest_rate * i_t_C
        dr_dt_D = self.params.gamma * i_t_D + self.params.arrest_rate * r_t_C + self.params.gamma_asymptom * e_t_D

        # community
        ds_dt_C = -self.params.sigma * self.params.beta_community * (s_t_C * e_t_C) - self.params.beta_community * (
                s_t_C * i_t_C) \
                  - self.params.sigma * self.params.beta_community * (s_t_C * e_t_O_C) - self.params.beta_community * (
                          s_t_C * i_t_O_C) \
                  - self.params.arrest_rate * s_t_C
        de_dt_C = self.params.sigma * self.params.beta_community * (s_t_C * e_t_C) + self.params.beta_community * (
                s_t_C * i_t_C) - self.params.gamma_ei * e_t_C \
                  + self.params.sigma * self.params.beta_community * (s_t_C * e_t_O_C) + self.params.beta_community * (
                          s_t_C * i_t_O_C) - self.params.arrest_rate * e_t_C - self.params.gamma_asymptom * e_t_C
        di_dt_C = -self.params.gamma * i_t_C + self.params.gamma_ei * e_t_C \
                  + self.params.alos * i_t_D - self.params.arrest_rate * i_t_C
        dr_dt_C = self.params.gamma * i_t_C - self.params.arrest_rate * r_t_C + self.params.gamma_asymptom * e_t_C

        # employees/officers
        ds_dt_O_C = -self.params.sigma * self.params.beta_community * (s_t_O_C * e_t_C) - self.params.beta_community * (
                s_t_O_C * i_t_C) \
                    + self.params.staff_work_shift * s_t_O_D - self.params.staff_home_shift * s_t_O_C
        de_dt_O_C = self.params.sigma * self.params.beta_community * (s_t_O_C * e_t_C) + self.params.beta_community * (
                s_t_O_C * i_t_C) - self.params.gamma_ei * e_t_O_C \
                    + self.params.staff_work_shift * e_t_O_D - self.params.staff_home_shift * e_t_O_C - self.params.gamma_asymptom * e_t_O_C
        di_dt_O_C = -self.params.gamma * i_t_O_C + self.params.gamma_ei * e_t_O_C \
                    + self.params.staff_work_shift * i_t_O_D - self.params.staff_home_shift * i_t_O_C
        dr_dt_O_C = self.params.gamma * i_t_O_C \
                    + self.params.staff_work_shift * r_t_O_D - self.params.staff_home_shift * r_t_O_C + self.params.gamma_asymptom * e_t_O_C

        ds_dt_O_D = -self.params.sigma * self.params.beta_detention * (
                s_t_O_D * e_t_O_D) - self.params.beta_detention * (s_t_O_D * i_t_O_D) \
                    - self.params.sigma * self.params.beta_detention * (
                            s_t_O_D * e_t_D) - self.params.beta_detention * (s_t_O_D * i_t_D) \
                    - self.params.staff_work_shift * s_t_O_D + self.params.staff_home_shift * s_t_O_C
        de_dt_O_D = self.params.sigma * self.params.beta_detention * (
                s_t_O_D * e_t_O_D) + self.params.beta_detention * (
                            s_t_O_D * i_t_O_D) - self.params.gamma_ei * e_t_O_D \
                    + self.params.sigma * self.params.beta_detention * (
                            s_t_O_D * e_t_D) + self.params.beta_detention * (s_t_O_D * i_t_D) \
                    - self.params.staff_work_shift * e_t_O_D + self.params.staff_home_shift * e_t_O_C \
                    - self.params.gamma_asymptom * e_t_O_D
        di_dt_O_D = -self.params.gamma * i_t_O_D + self.params.gamma_ei * e_t_O_D \
                    - self.params.staff_work_shift * i_t_O_D + self.params.staff_home_shift * i_t_O_C
        dr_dt_O_D = self.params.gamma * i_t_O_D \
                    - self.params.staff_work_shift * r_t_O_D + self.params.staff_home_shift * r_t_O_C \
                    + self.params.gamma_asymptom * e_t_O_D

        new_y = np.array([ds_dt_D, de_dt_D, di_dt_D, dr_dt_D,
                          ds_dt_C, de_dt_C, di_dt_C, dr_dt_C,
                          ds_dt_O_C, de_dt_O_C, di_dt_O_C, dr_dt_O_C,
                          ds_dt_O_D, de_dt_O_D, di_dt_O_D, dr_dt_O_D])
        return new_y


def solve_model(model_params=None, county_pop=10000, staff_pop=10, detention_pop=100, beta=2.43, sigma=0.5, gamma=1 / 10, gamma_ei=1 / 6.7,
          staff_work_shift=3, c_jail=3, c_0=500, init_community_infections=500, init_detention_infections=0,
          arrest_rate=.0000035, alos=.0167):
    # Should be able to set parameters with county (city) and ICE facility and staff population
    # Defaults:
    # beta = 0.625  # An infected person infects a person every 2 days (half a person per day) (this might be after social distancing so should increase)
    # beta = 2.43 # can be greater than 1 for rate in ODE model, adult risk from Lofgren paper
    ## beta is arbitrary and can be re-calibrated later
    # sigma = 0.5
    # gamma = 1/10  # Infectious period is 10 days
    # gamma_ei = 1/5.1  # latent period is 5 days, say
    # staff_work_shift = 3
    # c_jail = 3

    N = model_params.detention_pop + model_params.county_pop + model_params.staff_pop \
        + model_params.init_community_infections + model_params.init_detention_infections

    y_init = np.array([(model_params.detention_pop - model_params.init_detention_infections) / N, 0,
                       model_params.init_detention_infections / N, 0,
                       model_params.county_pop / N, 0, model_params.init_community_infections / N, 0,
                       (model_params.staff_pop / 2) / N, 0 / N, 0, 0,
                       (model_params.staff_pop / 2) / N, 0 / N, 0, 0])  # split staff  between in community and facility

    if model_params is None:
        model_params = ModelParams(beta, sigma, gamma, gamma_ei, staff_work_shift, c_jail, arrest_rate, alos,
                                   county_pop, staff_pop, detention_pop, c_0, init_community_infections,
                                   init_detention_infections)
    model_params.callibrate()
    calibrated_model = Model(model_params, N, y_init)
    solution_ts = calibrated_model.solve_model()
    print(f'Beta community/detention Params: {model_params.beta_community}, {model_params.beta_detention}')

    return solution_ts
