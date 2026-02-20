from .Emcee_utils import ensemble_sampler, hDFBackend_2
import time
from scipy.optimize import minimize
from . import generate_data_utils, Exp_data_utils
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

def get_maximum_likelihood_estimate(
    Exp_data_el,
    Exp_data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    model_config,
    save_folder,
    coeff_spread=10,
    num_coords=32,
    fixed_parameters_dict={},
    params_to_fit={},
    min_bound={},
    max_bound={},
):
    nll = lambda *args: -generate_data_utils.el_loglike(*args)
    init_params, min_bound_list, max_bound_list = [], [], []
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit[key] == {}:
            continue
        for key2 in params_to_fit[key].keys():
            init_params.append(params_to_fit[key][key2])
            min_bound_list.append(min_bound[key][key2])
            max_bound_list.append(max_bound[key][key2])
            counter += 1
    min_bound_list = np.array(min_bound_list)
    max_bound_list = np.array(max_bound_list)
    num_parameters = counter

    coords = init_params + 0.1 * coeff_spread * (
        max_bound_list - min_bound_list
    ) * np.random.randn(num_coords, num_parameters)
    # coords = init_params + coeff_spread * (
    # max_bound_list - min_bound_list
    # ) * np.random.uniform(-1, 1, (num_coords, num_parameters))
    
    min_fun = np.inf
    print("running the minimisation")
    for i, coord in enumerate(coords):
        print(f"step {i}")
        soln = minimize(
            nll,
            coord,
            args=(
                Exp_data_el,
                Exp_data_pl,
                co_var_mat_el,
                co_var_mat_pl,
                model_config["temperature_list_el"],
                model_config["hws_el"],
                model_config["temperature_list_pl"],
                model_config["hws_pl"],
                fixed_parameters_dict,
                params_to_fit,
            ),
            bounds=[
                (min_bound_list[i], max_bound_list[i])
                for i in range(num_parameters)
            ],
            tol=1e-2,
        )
        if "NORM_OF_PROJECTED_GRADIENT_" in soln.message:
            print("NORM_OF_PROJECTED_GRADIENT_")
            print(soln.x)
            print(soln.fun)
            continue
        if soln.fun < min_fun:
            min_fun = soln.fun
            soln_min = soln
    print(soln_min.x)
    print("Maximum likelihood estimates:")
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit[key] == {}:
            continue
        for key2 in params_to_fit[key].keys():
            print(f"  {key}_{key2} = {soln_min.x[counter]:.3f}")
            counter += 1
    print("Maximum log likelihood:", soln.fun)
    # print those into a file
    with open(save_folder + "/maximum_likelihood_estimate.txt", "w") as f:
        f.write("Maximum likelihood estimates:\n")
        counter = 0
        for key in ["EX", "CT", "D"]:
            if params_to_fit[key] == {}:
                continue
            for key2 in params_to_fit[key].keys():
                f.write(f"  {key}_{key2} = {soln_min.x[counter]:.3f}")
                counter += 1

        f.write(f"Maximum log likelihood: {soln_min.fun}\n")
    return soln_min

def run_sampler_single(
    save_folder,
    Exp_data_el,
    Exp_data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    params_to_fit,
    fixed_parameters_dict,
    min_bound,
    max_bound,
    model_config,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
):
    init_params, min_bound_list, max_bound_list = [], [], []
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit[key] == {}:
            continue
        for key2 in params_to_fit[key].keys():
            init_params.append(params_to_fit[key][key2])
            min_bound_list.append(min_bound[key][key2])
            max_bound_list.append(max_bound[key][key2])
            counter += 1
    min_bound_list = np.array(min_bound_list)
    max_bound_list = np.array(max_bound_list)
    num_parameters = counter

    coords = init_params + 0.1 * coeff_spread * (
        max_bound_list - min_bound_list
    ) * np.random.randn(num_coords, num_parameters)

    # coords = init_params + coeff_spread * (
    # max_bound_list - min_bound_list
    # ) * np.random.uniform(-1, 1, (num_coords, num_parameters))
    
    nwalkers, ndim = coords.shape
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = save_folder + "/sampler.h5"
    backend = hDFBackend_2(filename, name="single_core")
    backend.reset(nwalkers, ndim)
    print("Initial size: {0}".format(backend.iteration))

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf

    # Here are the important lines

    sampler = ensemble_sampler(
        nwalkers,
        ndim,
        generate_data_utils.log_probability,
        args=(
            Exp_data_el,
            Exp_data_pl,
            co_var_mat_el,
            co_var_mat_pl,
            model_config,
            fixed_parameters_dict,
            params_to_fit,
            min_bound,
            max_bound,
        ),
        backend=backend,

    )
    start = time.time()
    blobs = []
    for sample in sampler.sample(
        coords, iterations=nsteps, progress=True, blobs0=blobs
    ):
        if sampler.iteration % 100:
            continue
        try:
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            print(tau)
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            end = time.time()
        except Exception as e:
            print(e)
            print("error in the autocorrelation time")
    # sampler.sample(pos, iterations = nsteps, progress=True,store=True)
    end = time.time()
    multi_time = end - start
    print("single process took {0:.1f} seconds".format(multi_time))
    # print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    return sampler


def run_sampler_parallel(
    save_folder,
    Exp_data_el,
    Exp_data_pl,
    co_var_mat_el,
    co_var_mat_pl,
    params_to_fit,
    fixed_parameters_dict,
    min_bound,
    max_bound,
    model_config,
    nsteps=10000,
    coeff_spread=10,
    num_coords=32,
):
    init_params, min_bound_list, max_bound_list = [], [], []
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit[key] == {}:
            continue
        for key2 in params_to_fit[key].keys():
            init_params.append(params_to_fit[key][key2])
            min_bound_list.append(min_bound[key][key2])
            max_bound_list.append(max_bound[key][key2])
            counter += 1
    min_bound_list = np.array(min_bound_list)
    max_bound_list = np.array(max_bound_list)
    num_parameters = counter

    coords = init_params + 0.1 * coeff_spread * (
        max_bound_list - min_bound_list
    ) * np.random.randn(num_coords, num_parameters)

    # coords = init_params + coeff_spread * (
    # max_bound_list - min_bound_list
    # ) * np.random.uniform(-1, 1, (num_coords, num_parameters))
    
    nwalkers, ndim = coords.shape
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = save_folder + "/sampler.h5"
    backend = hDFBackend_2(filename, name="single_core")
    backend.reset(nwalkers, ndim)
    print("Initial size: {0}".format(backend.iteration))

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)
    # This will be useful to testing convergence
    old_tau = np.inf

    # Here are the important lines
    with Pool() as pool:
        sampler = ensemble_sampler(
            nwalkers,
            ndim,
            generate_data_utils.log_probability,
            args=(
                Exp_data_el,
                Exp_data_pl,
                co_var_mat_el,
                co_var_mat_pl,
                model_config,
                fixed_parameters_dict,
                params_to_fit,
                min_bound,
                max_bound,
            ),
            backend=backend,
            pool=pool,

        )
        start = time.time()
        blobs = []
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(
            coords, iterations=nsteps, progress=True, blobs0=blobs
        ):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                print(tau)
                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
                end = time.time()
            except Exception as e:
                print(e)
                print("error in the autocorrelation time")
        # sampler.sample(pos, iterations = nsteps, progress=True,store=True)
        end = time.time()
        multi_time = end - start
        print("single process took {0:.1f} seconds".format(multi_time))
        # print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    return sampler


# def plot_exp_data_with_variance(
#     temperature_list_el,
#     hws_el,
#     temperature_list_pl,
#     hws_pl,
#     variance_el,
#     variance_pl,
#     save_folder,
#     fixed_parameters_dict,
#     true_parameters,
#     Exp_data_pl,
#     Exp_data_el,
# ):
    
#     model_data_el, model_data_pl = generate_data_utils.el_trial(
#         temperature_list_el,
#         hws_el,
#         temperature_list_pl,
#         hws_pl,
#         fixed_parameters_dict,
#         true_parameters,
#     )
#     truemodel_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
#     truemodel_el = model_data_el / np.max(model_data_el.reshape(-1, 1))

    
#     fig, axis = Exp_data_utils.plot_pl_data_with_variance(
#         Exp_data_pl, temperature_list_pl, hws_pl, variance_pl, save_folder, dpi=300
#     )
#     for i, axes in enumerate(axis):
#         colors = ["#8297B7", "#63BBB5", "#ECC79F", "#DC7600", "#7D063D"]
#         axes.plot(hws_pl, truemodel_pl[:, i], label="Cal.", color=colors[i], linewidth=1.8)
#         # axes.plot(hws_pl, truemodel_pl[:, i], label="fit", color="C" + str(i))
#         axes.legend(fontsize=15, frameon=False)

#         axes.tick_params(axis='y', labelsize=11)
#         axes.tick_params(axis='x', labelsize=0, direction='in')
#         axes.set_xticklabels([])

#         axes.set_title("Temperature="+str(temperature_list_pl[i])+" K", fontsize=18)
#         axes.set_xlim(0.8, 1.8)
#         axes.set_ylim(0, 1.1)
#         axes.set_aspect(10/11)
#         # axes.set_xlabel("Photon Energy (eV)", fontsize=18)
#         axes.set_ylabel("PL Intensity (a.u.)", fontsize=18)
#     fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     fig.tight_layout(h_pad=0.0)

    # fig, axis = Exp_data_utils.plot_pl_data_with_variance(
#         Exp_data_el, temperature_list_el, hws_el, variance_el, save_folder, dpi=300
#     )
#     for i, axes in enumerate(axis):
#         colors = ["#8297B7", "#63BBB5", "#ECC79F", "#DC7600", "#7D063D"]
#         axes.plot(hws_el, truemodel_el[:, i], label="Cal.", color=colors[i], linewidth=1.8)
#         # axes.plot(hws_el, truemodel_el[:, i], label="fit", color="C" + str(i))
#         axes.legend(fontsize=15, frameon=False)
#         axes.tick_params(axis='both', labelsize=11)
#         axes.set_xlim(0.8, 1.8)
#         axes.set_ylim(0, 1.1)
#         axes.set_aspect(10/11)
#         axes.set_xlabel("Photon Energy (eV)", fontsize=18)
#         axes.set_ylabel("EL intensity (a.u.)", fontsize=18)
#     fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     fig.tight_layout(h_pad=0.0)
#     return fig, axis

def plot_exp_data_with_variance(
    temperature_list_el,
    hws_el,
    temperature_list_pl,
    hws_pl,
    variance_el,
    variance_pl,
    save_folder,
    fixed_parameters_dict,
    true_parameters,
    Exp_data_pl,
    Exp_data_el,
):
    model_data_el, model_data_pl = generate_data_utils.el_trial(
        temperature_list_el,
        hws_el,
        temperature_list_pl,
        hws_pl,
        fixed_parameters_dict,
        true_parameters,
    )
    truemodel_pl = model_data_pl / np.max(model_data_pl.reshape(-1, 1))
    truemodel_el = model_data_el / np.max(model_data_el.reshape(-1, 1))

    fig, axes = plt.subplots(2, len(temperature_list_pl), figsize=(24, 9), dpi=300)

    Exp_data_pl = Exp_data_pl/max(Exp_data_pl.reshape(-1, 1))
    for i in range(len(temperature_list_pl)):
        ax = axes[0, i]
        colors = ["#8297B7", "#63BBB5", "#ECC79F", "#DC7600", "#7D063D"]

        exp_line = ax.plot(
            hws_pl,
            Exp_data_pl[:, i],
            label="Exp.",
            linestyle="--",
            color=colors[i],
            alpha=0.5,
            linewidth=2,
        )[0]
        
        ax.fill_between(
            hws_pl,
            Exp_data_pl[:, i] - np.sqrt(variance_pl[:, i]),
            Exp_data_pl[:, i] + np.sqrt(variance_pl[:, i]),
            alpha=0.3,
            color=colors[i],
        )

        cal_line = ax.plot(hws_pl, truemodel_pl[:, i], label="Cal.", color=colors[i], linewidth=1.8)[0]

        # Check upper right region data values
        right_upper_idx = np.where((hws_pl > 1.45) & (hws_pl < 1.8))[0]
        max_value_right_upper = 0
        if len(right_upper_idx) > 0:
            max_value_right_upper = max(np.max(Exp_data_pl[right_upper_idx, i]), 
                                        np.max(truemodel_pl[right_upper_idx, i]))
        
        # Determine legend location based on upper right data values
        legend_loc = 'upper left' if max_value_right_upper > 0.7 else 'upper right'
        legend_bbox = (0.05, 0.95) if max_value_right_upper > 0.7 else (1, 0.95)
        
        legend1 = ax.legend([exp_line, cal_line], ["Exp.", "Cal."], 
                          fontsize=15, frameon=False, loc=legend_loc,
                          bbox_to_anchor=legend_bbox)

        ax.add_artist(legend1)

        from matplotlib.lines import Line2D
        empty_line = Line2D([0], [0], color=colors[i], marker='', linestyle='')
        
        # Temperature legend uses the same horizontal position
        temp_legend_bbox = (0.05, 1.02) if max_value_right_upper > 0.7 else (1, 1.02)
        ax.legend([empty_line], ["@ " + str(temperature_list_pl[i]) + " K"], 
                 loc=legend_loc, bbox_to_anchor=temp_legend_bbox, 
                 fontsize=15, frameon=False, handlelength=0, handletextpad=0)
        
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=0, length=4, direction='in')
        ax.set_xticklabels([])
        ax.set_xlim(0.8, 1.8)
        ax.set_ylim(0, 1.1)
        ax.set_aspect(10/11)
        ax.set_ylabel("PL Intensity (a.u.)", fontsize=18)

    Exp_data_el = Exp_data_el/max(Exp_data_el.reshape(-1, 1))
    for i in range(len(temperature_list_el)):
        ax = axes[1, i]
        colors = ["#8297B7", "#63BBB5", "#ECC79F", "#DC7600", "#7D063D"]

        exp_line = ax.plot(
            hws_el,
            Exp_data_el[:, i],
            label="Exp.",
            linestyle="--",
            color=colors[i],
            alpha=0.5,
            linewidth=2,
        )[0]
        
        ax.fill_between(
            hws_el,
            Exp_data_el[:, i] - np.sqrt(variance_el[:, i]),
            Exp_data_el[:, i] + np.sqrt(variance_el[:, i]),
            alpha=0.3,
            color=colors[i],
        )

        cal_line = ax.plot(hws_el, truemodel_el[:, i], label="Cal.", color=colors[i], linewidth=1.8)[0]

        # Check upper right region data values
        right_upper_idx = np.where((hws_el > 1.45) & (hws_el < 1.8))[0]
        max_value_right_upper = 0
        if len(right_upper_idx) > 0:
            max_value_right_upper = max(np.max(Exp_data_el[right_upper_idx, i]), 
                                        np.max(truemodel_el[right_upper_idx, i]))
        
        # Determine legend location based on upper right data values
        legend_loc = 'upper left' if max_value_right_upper > 0.7 else 'upper right'
        legend_bbox = (0.05, 0.95) if max_value_right_upper > 0.7 else (1, 0.95)
        
        legend1 = ax.legend([exp_line, cal_line], ["Exp.", "Cal."], 
                          fontsize=15, frameon=False, loc=legend_loc,
                          bbox_to_anchor=legend_bbox)

        ax.add_artist(legend1)

        from matplotlib.lines import Line2D
        empty_line = Line2D([0], [0], color=colors[i], marker='', linestyle='')
        
        # Temperature legend uses the same horizontal position
        temp_legend_bbox = (0.05, 1.02) if max_value_right_upper > 0.7 else (1, 1.02)
        ax.legend([empty_line], ["@ " + str(temperature_list_el[i]) + " K"], 
                 loc=legend_loc, bbox_to_anchor=temp_legend_bbox, 
                 fontsize=15, frameon=False, handlelength=0, handletextpad=0)
        
        ax.tick_params(axis='both', labelsize=11)
        ax.set_xlim(0.8, 1.8)
        ax.set_ylim(0, 1.1)
        ax.set_aspect(10/11)
        ax.set_xlabel("Photon Energy (eV)", fontsize=18)
        ax.set_ylabel("EL intensity (a.u.)", fontsize=18)
        ax.spines['top'].set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout(h_pad=-2.5)  # h_pad set to negative value for closer spacing
    
    return fig, axes

def get_param_dict(params_to_fit_init,true_params_list):

    true_parameters = {
        "EX": {},
        "CT": {},
        "D": {},
    }
    counter = 0
    for key in ["EX", "CT", "D"]:
        if params_to_fit_init[key] == {}:
            continue
        for id, key2 in enumerate(params_to_fit_init[key].keys()):
            true_parameters[key][key2] = true_params_list[counter]
            counter += 1
    return true_parameters
