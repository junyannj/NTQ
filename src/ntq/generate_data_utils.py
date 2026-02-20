from scipy.optimize import minimize
import numpy as np
from .model_function import LTL
import os


def generate_data(
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    Temp_std_err,
    hws_std_err,
    relative_intensity_std_error_pl,
    relative_intensity_std_error_el,
    sigma,
    params_to_fit={},
    fixed_parameters_dict={},
):
    """Generate the data for the el and pl spectra with added noise

    Args:

            temperature_list_el (np.array): The temperature list for the el spectra
            hws_el (np.array): The photon energies for the el spectra
            temperature_list_pl (np.array): The temperature list for the pl spectra
            hws_pl (np.array): The photon energies for the pl spectra
            Temp_std_err (float): The standard deviation of the temperature error
            hws_std_err (float): The standard deviation of the photon energy error
            relative_intensity_std_error_pl (float): The standard deviation of the relative intensity error for the pl spectra
            relative_intensity_std_error_el (float): The standard deviation of the relative intensity error for the el spectra
            number_free_parameters (int): The number of free parameters in the model
            params_to_fit (dict): The parameters to fit in the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            tuple: The model data for the pl and el spectra and the true parameters
    """

    # relative intensity error
    def add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error
    ):
        relative_intensity_model = np.max(model_data_pl, axis=0) / max(
            model_data_pl.reshape(-1, 1)
        )
        relative_intensity_model_error = (
            relative_intensity_model
            + np.random.normal(
                0, relative_intensity_std_error, len(relative_intensity_model)
            )
        )
        relative_intensity_model_error = np.abs(
            relative_intensity_model_error
            / np.max(relative_intensity_model_error)
        )
        model_data_pl = (
            model_data_pl
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data_pl

    # error in the temperature of the sample

    temperature_list_pl = temperature_list_pl + np.random.normal(
        0, Temp_std_err, len(temperature_list_pl)
    )
    temperature_list_el = temperature_list_el + np.random.normal(
        0, Temp_std_err, len(temperature_list_el)
    )
    # error in the detection wavelength
    hws_pl = hws_pl + np.random.normal(0, hws_std_err, len(hws_pl))
    hws_el = hws_el + np.random.normal(0, hws_std_err, len(hws_el))
    model_data_el, model_data_pl = el_trial(
        temperature_list_el,
        hws_el,
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        params_to_fit,
    )
    model_data_pl = add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error_pl
    )
    model_data_el = add_relative_intensity_error(
        model_data_el, relative_intensity_std_error_el
    )

    return model_data_pl, model_data_el


def generate_data_pl(
    temperature_list_pl,
    hws_pl,
    Temp_std_err,
    hws_std_err,
    relative_intensity_std_error_pl,
    sigma,
    params_to_fit={},
    fixed_parameters_dict={},
    **kwargs,
):
    """Generate the data for the pl spectra with added noise

    Args:


            temperature_list_pl (np.array): The temperature list for the pl spectra
            hws_pl (np.array): The photon energies for the pl spectra
            Temp_std_err (float): The standard deviation of the temperature error
            hws_std_err (float): The standard deviation of the photon energy error
            relative_intensity_std_error_pl (float): The standard deviation of the relative intensity error for the pl spectra
            number_free_parameters (int): The number of free parameters in the model
            params_to_fit (dict): The parameters to fit in the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            tuple: The model data for the pl and el spectra and the true parameters
    """

    # relative intensity error
    def add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error
    ):
        relative_intensity_model = np.max(model_data_pl, axis=0) / max(
            model_data_pl.reshape(-1, 1)
        )
        relative_intensity_model_error = (
            relative_intensity_model
            + np.random.normal(
                0, relative_intensity_std_error, len(relative_intensity_model)
            )
        )
        relative_intensity_model_error = np.abs(
            relative_intensity_model_error
            / np.max(relative_intensity_model_error)
        )
        model_data_pl = (
            model_data_pl
            * relative_intensity_model_error
            / relative_intensity_model
        )
        return model_data_pl

    # error in the temperature of the sample

    temperature_list_pl = temperature_list_pl + np.random.normal(
        0, Temp_std_err, len(temperature_list_pl)
    )
    # error in the detection wavelength
    hws_pl = hws_pl + np.random.normal(0, hws_std_err, len(hws_pl))
    model_data_pl, EX_kr, Ex_knr = pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        params_to_fit,
    )
    model_data_pl = add_relative_intensity_error(
        model_data_pl, relative_intensity_std_error_pl
    )
    return model_data_pl, EX_kr, Ex_knr 


def set_parameters(data, fixed_parameters_dict):
    """Set the fixed parameters for the model that are not the same as the default

    Args:
            data (LTL.Data): The data object for the model
            fixed_parameters_dict (dict): The fixed parameters for the model in a dictionary for the different classes
    Returns:
            LTL.Data: The data object for the model with the fixed parameters set
    """
    for key, value in fixed_parameters_dict.items():
        data[key].update(value)
    return data


def pl_trial(
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Run the model for the  pl spectra"""
    data = LTL.Data()
    data.update(**fixed_parameters_dict)
    data.update(**params_to_fit)
    data.D.Luminecence_exp = "PL"
    data.D.T = temperature_list_pl  # np.array([300.0, 150.0, 80.0])
    LTL.LTLCalc(data)
    pl_results = data.D.kr_hw  # .reshape(-1, 1)
    pl_results_interp = np.zeros((len(hws_pl), len(temperature_list_pl)))
    for i in range(len(temperature_list_pl)):
        pl_results_interp[:, i] = np.interp(
            hws_pl, data.D.hw, pl_results[:, i]
        )
    EX_kr = data.EX.kr
    Ex_knr = data.EX.knr
    return pl_results_interp, EX_kr, Ex_knr 


def el_trial(
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    """Run the model for the el and pl spectra"""
    data = LTL.Data()
    # print(f"fixed_parameters_dict is {fixed_parameters_dict}")
    # print(f"params_to_fit is {params_to_fit}")
    data.update(**fixed_parameters_dict)
    data.update(**params_to_fit)
    data.D.Luminecence_exp = "EL"
    LTL.LTLCalc(data)
    el_results = data.D.kr_hw  # .reshape(-1, 1)
    el_results_interp = np.zeros((len(hws_el), len(temperature_list_el)))
    for i in range(len(temperature_list_el)):
        el_results_interp[:, i] = np.interp(
            hws_el, data.D.hw, el_results[:, i]
        )
    data.D.Luminecence_exp = "PL"
    data.D.T = temperature_list_pl  # np.array([300.0, 150.0, 80.0])
    LTL.LTLCalc(data)
    pl_results = data.D.kr_hw  # .reshape(-1, 1)
    pl_results_interp = np.zeros((len(hws_pl), len(temperature_list_pl)))
    for i in range(len(temperature_list_pl)):
        pl_results_interp[:, i] = np.interp(
            hws_pl, data.D.hw, pl_results[:, i]
        )
    return el_results_interp, pl_results_interp  # / max(pl_results)


def log_prior(theta, min_bounds, max_bounds):
    counter = 0
    for param_key in ["EX", "CT", "D"]:
        if min_bounds[param_key] == {}:
            continue

        for id, key in enumerate(min_bounds[param_key].keys()):
            if (
                min_bounds[param_key][key] > theta[counter]
                or max_bounds[param_key][key] < theta[counter]
            ):
                return -np.inf
            counter += 1
    return 0.0


def el_loglike(
    theta,
    data_el,
    data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    params_to_fit_updated = {"EX": {}, "CT": {}, "D": {}}
    counter = 0
    try:
        for key in ["EX", "CT", "D"]:
            if params_to_fit[key] == {}:
                continue

            for id, key2 in enumerate(params_to_fit[key].keys()):
                params_to_fit_updated[key][key2] = theta[counter]
                counter += 1

    except Exception as e:
        print(e)
        raise ValueError("The parameters to fit are not in the correct format")

    model_data_el, model_data_pl = el_trial(
        temperature_list_el,
        hws_el,
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        params_to_fit_updated,
    )
    model_data_el = model_data_el / np.max(model_data_el.reshape(-1, 1))
    model_data_el = model_data_el.reshape(-1, 1)
    data_el = data_el / np.max(data_el.reshape(-1, 1))
    data_el = data_el.reshape(-1, 1)
    model_data_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    model_data_pl = model_data_pl.reshape(-1, 1)
    data_pl = data_pl / np.max(data_pl.reshape(-1, 1))
    data_pl = data_pl.reshape(-1, 1)
    # check that the data in model_data does not contain NaNs or infs
    if np.isnan(model_data_el).any() or np.isinf(model_data_el).any():
        # print("NaN in model_data")
        return -np.inf
    diff_el = data_el - model_data_el
    diff_el[np.abs(diff_el) < 1e-3] = 0
    diff_el[np.abs(data_el) < 3e-2] = 0
    loglike = -0.5 * np.dot(
        diff_el.T, np.dot(np.linalg.inv(co_var_mat_el), diff_el)
    )
    diff_pl = data_pl - model_data_pl
    diff_pl[np.abs(diff_pl) < 1e-3] = 0
    diff_pl[np.abs(data_pl) < 3e-2] = 0
    loglike = loglike - 0.5 * np.dot(
        diff_pl.T, np.dot(np.linalg.inv(co_var_mat_pl), diff_pl)
    )
    return loglike


def log_probability(
    theta,
    data_el,
    data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    X,
    fixed_parameters_dict,
    params_to_fit,
    min_bounds,
    max_bounds,
):
    lp = log_prior(theta, min_bounds, max_bounds)
    if lp == -np.inf:
        return -np.inf
    log_like = el_loglike(
        theta,
        data_el,
        data_pl,
        co_var_mat_el,
        co_var_mat_pl,
        X["temperature_list_el"],
        X["hws_el"],
        X["temperature_list_pl"],
        X["hws_pl"],
        fixed_parameters_dict,
        params_to_fit,
    )
    
    # if log_like is None or np.isnan(log_like[0]) or np.isinf(log_like[0]):
    #     return -np.inf
    # log_prob = lp + log_like[0]
    # if np.isnan(log_prob) or np.isinf(log_prob):
    #     return -np.inf
    # assert (
    #     isinstance(log_prob, float)
    # ), f"The log_prob is not a float but a {type(log_prob)}"

    log_prob = lp + log_like[0]
    # print(f"log_prob is {log_prob}")
    # assert log_prob is a float
    if np.isnan(log_like):
        return -np.inf
    if np.isinf(log_like):
        return -np.inf
    if log_prob is None:
        return -np.inf
    assert (
        log_prob.dtype.kind == "f"
    ), f"the log_prob is not a float but a {type(log_prob)}"

    return log_prob


def pl_loglike(
    theta,
    data_pl,
    co_var_mat_pl,
    temperature_list_pl,
    hws_pl,
    fixed_parameters_dict={},
    params_to_fit={},
):
    params_to_fit_updated = {"EX": {}, "CT": {}, "D": {}}
    counter = 0
    try:
        for key in ["EX", "CT", "D"]:
            if params_to_fit[key] == {}:
                continue

            for id, key2 in enumerate(params_to_fit[key].keys()):
                params_to_fit_updated[key][key2] = theta[counter]
                counter += 1

    except Exception as e:
        print(e)
        raise ValueError("The parameters to fit are not in the correct format")
    model_data_pl, EX_kr, Ex_knr  = pl_trial(
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        params_to_fit_updated,
    )
    model_data_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    model_data_pl = model_data_pl.reshape(-1, 1)
    data_pl = data_pl / np.max(data_pl.reshape(-1, 1))
    data_pl = data_pl.reshape(-1, 1)
    diff_pl = data_pl - model_data_pl
    diff_pl[np.abs(data_pl) < 3e-2] = 0
    loglike = -0.5 * np.dot(
        diff_pl.T, np.dot(np.linalg.inv(co_var_mat_pl), diff_pl)
    )
    Chi_squared = np.dot(
        diff_pl.T, np.dot(np.linalg.inv(co_var_mat_pl), diff_pl)
    ) / (len(data_pl) - len(theta))
    return loglike, Chi_squared, EX_kr, Ex_knr 


def log_probability_pl(
    theta,
    data_pl,
    co_var_mat_pl,
    X,
    fixed_parameters_dict,
    params_to_fit,
    min_bounds,
    max_bounds,
):
    lp = log_prior(theta, min_bounds, max_bounds)
    if lp == -np.inf:
        return -np.inf, None, None, None, None
    log_like, Chi_squared, EX_kr, Ex_knr  = pl_loglike(
        theta,
        data_pl,
        co_var_mat_pl,
        X["temperature_list_pl"],
        X["hws_pl"],
        fixed_parameters_dict,
        params_to_fit,
    )

    # if log_like is None or np.isnan(log_like[0]) or np.isinf(log_like[0]):
    #     return -np.inf, None, None, None, None
    # log_prob = lp + log_like[0]
    # if np.isnan(log_prob) or np.isinf(log_prob):
    #     return -np.inf, None, None, None, None
    # assert isinstance(log_prob, float), f"The log_prob is not a float but a {type(log_prob)}"

    log_prob = lp + log_like[0]
    # print(f"log_prob is {log_prob}")
    # assert log_prob is a float
    if np.isnan(log_like):
        return -np.inf, None, None, None, None
    if np.isinf(log_like):
        return -np.inf, None, None, None, None
    if log_prob is None:
        return -np.inf, None, None, None, None
    assert (
        log_prob.dtype.kind == "f"
    ), f"the log_prob is not a float but a {type(log_prob)}"

    return log_prob, log_like[0], Chi_squared, EX_kr[-1], Ex_knr[0][-1] 
