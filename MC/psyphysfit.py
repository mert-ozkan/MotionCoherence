from __future__ import absolute_import, division, print_function
import numpy as np
import bayesfit as bf
from bayesfit.psyFunction import psyfunction


def get_parameter_estimates(options, metrics, estimate_type='MAP'):
    """
    Copied from plot_psyFcn.py
    """

    param_guess = np.zeros(4)
    counter = 0
    for keys in options['param_free']:
        if keys is True:
            param_guess[counter] = metrics[estimate_type][counter]
        elif keys is False:
            param_guess[counter] = options['param_ests'][counter]
        counter += 1

    return list(param_guess)


def return_predictedfunction(x_dat,
                             param_estx,
                             sigmoid_type='logistic',
                             x_lim='proportion',
                             x_precision=1000):
    if x_lim == 'minmax':
        x_min = min(x_dat)
        x_max = max(x_dat)
    elif x_lim == 'proportion':
        x_min = 0
        x_max = 1
    elif x_lim == 'percentage':
        x_min = 0
        x_max = 100
    else:
        x_min = x_lim[0]
        x_max = x_lim[1]

    x_est = list(np.linspace(x_min,
                             x_max,
                             x_precision
                             )
                 )
    y_est = list(psyfunction(x_est,
                             param_estx[0],
                             param_estx[1],
                             param_estx[2],
                             param_estx[3],
                             sigmoid_type
                             )
                 )
    return {'x': x_est, 'y': y_est}


def psyfit_estimates(dat,
                     nafc=2,
                     sigmoid_type='logistic',
                     threshold=.75,
                     estimate_type='MAP',
                     est_x_lim='proportion',
                     est_x_precision=1000):
    metrics, options = bf.fitmodel(dat,
                                   nafc=nafc,
                                   sigmoid_type=sigmoid_type,
                                   threshold=threshold
                                   )
    threshold_est = bf.get_threshold(dat,
                                     options,
                                     metrics,
                                     threshold_pc=threshold,
                                     estimate_type=estimate_type
                                     )
    
    param_estx = get_parameter_estimates(options, metrics)

    est_dat = return_predictedfunction(dat[:, 0],
                                       param_estx,
                                       sigmoid_type=sigmoid_type,
                                       x_lim=est_x_lim,
                                       x_precision=est_x_precision
                                       )

    return {'est': est_dat,
            'threshold': threshold_est,
            'parameters': param_estx,
            'options': {'nafc': nafc,
                        'sigmoid_type': sigmoid_type,
                        'threshold': threshold,
                        'estimate_type': estimate_type,
                        'est_x_lim': est_x_lim,
                        'est_x_precision': est_x_precision
                        }
            }
