from __future__ import absolute_import, division, print_function
import numpy as np
import bayesfit as bf
import altair as alt
import pandas as pd
from functools import reduce
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


class PsyPhysFitObject:
    # est should be outputted from ppf.psyfit_estimates
    # dat should be Nx3 array (Intensity, qOk, qTrl)
    def __init__(self, dat, est):
        self.x_est = est['est']['x']
        self.y_est = est['est']['y']
        self.threshold = est['threshold']
        self.param_est = est['parameters']
        self.fitproperties = est['options']
        self.data = np.transpose(
            np.array([
                dat['Intensity'],
                dat['qOk'],
                dat['qTrl']
            ])
        )

    def getFitProperties(self):
        return self.fitproperties

    def getThreshold(self):
        return self.fitproperties['threshold'], self.threshold

    def __str__(self):
        return 'PsyPhysFitObject'


def encode_chart(chart,
                 x_label,
                 y_label,
                 x_lim,
                 y_lim,
                 c_label,
                 color_scheme,
                 title
                 ):
    if c_label is None:
        encoded_chart = chart.encode(alt.X(x_label,
                                           scale=alt.Scale(domain=x_lim)),
                                     alt.Y(y_label,
                                           scale=alt.Scale(domain=y_lim)))
    else:
        encoded_chart = chart.encode(alt.X(x_label,
                                           scale=alt.Scale(domain=x_lim)),
                                     alt.Y(y_label,
                                           scale=alt.Scale(domain=y_lim)),
                                     color=alt.Color(c_label + ':N', scale=alt.Scale(scheme=color_scheme)))

    if title is not None:
        encoded_chart = encoded_chart.properties(title=title)
    return encoded_chart


def makePsyPhysFitPlotData(obj,
                           x_label='Stimulus Intensity',
                           y_label='Proportion Correct',
                           c_label=None,
                           c_levels=None,
                           include_threshold=False,
                           min_y=0
                           ):
    dat_curve = pd.DataFrame({})
    dat_ptx = pd.DataFrame({})
    dat_threshold = pd.DataFrame({})

    if type(obj) == tuple or type(obj) == list:

        if c_label is None:
            c_label = 'Color'

        if c_levels is None:
            c_levels = list(range(len(obj)))

        assert type(c_levels) is list or type(c_levels) is tuple, \
            '''
            'c_levels' should be of types list or tuple.
            '''

        assert len(c_levels) == len(obj), \
            '''
            Elements in 'c_levels' should correspond to each PsyPhysFitObj.
            '''

        for idx in range(len(obj)):
            whObj = obj[idx]
            whC = c_levels[idx]
            currDatCurve = pd.DataFrame({x_label: whObj.x_est,
                                         y_label: whObj.y_est})

            currDatCurve[c_label] = whC

            currDatPtx = pd.DataFrame({x_label: whObj.data[:, 0],
                                       y_label: whObj.data[:, 1] / whObj.data[:, 2]})

            currDatPtx[c_label] = whC

            if include_threshold:
                threshold = whObj.getThreshold()

                currDatThreshold = pd.DataFrame({x_label: np.linspace(threshold[1], threshold[1], 50),
                                                 y_label: np.linspace(min_y, threshold[0], 50)})
                currDatThreshold[c_label] = whC
                dat_threshold = dat_threshold.append(currDatThreshold)
            dat_curve = dat_curve.append(currDatCurve)
            dat_ptx = dat_ptx.append(currDatPtx)

    #         elif type(obj)==Psy:
    else:
        dat_curve = pd.DataFrame({x_label: obj.x_est,
                                  y_label: obj.y_est})

        dat_ptx = pd.DataFrame({x_label: obj.data[:, 0],
                                y_label: obj.data[:, 1] / obj.data[:, 2]})

        if include_threshold:
            threshold = obj.getThreshold()
            dat_threshold = pd.DataFrame({x_label: np.linspace(threshold[1], threshold[1], 50),
                                          y_label: np.linspace(min_y, threshold[0], 50)})

    return dat_curve, dat_ptx, dat_threshold


def plot_psyphysfit(obj,
                    x_label='Stimulus Intensity',
                    y_label='Proportion Correct',
                    x_lim=[0, 1],
                    y_lim=[.5, 1],
                    c_label=None,
                    c_levels=None,
                    include_threshold=True,
                    include_datapoints=True,
                    color_scheme='dark2',
                    title=None
                    ):
    x_lim = tuple(x_lim)
    y_lim = tuple(y_lim)

    dat = makePsyPhysFitPlotData(obj,
                                 x_label=x_label, y_label=y_label,
                                 c_label=c_label, c_levels=c_levels,
                                 include_threshold=include_threshold, min_y=y_lim[0])

    dat_curve, dat_ptx, dat_threshold = dat

    plt_curve = alt.Chart(dat_curve).mark_line()

    plt_ptx = alt.Chart(dat_ptx).mark_point()

    plt_threshold = alt.Chart(dat_threshold).mark_point(filled=True, size=10)

    if c_label is None and (type(obj) == tuple or type(obj) == list):
        c_label = 'Color'

    pltx = [plt_curve]
    if include_datapoints:
        pltx.append(plt_ptx)
    if include_threshold:
        pltx.append(plt_threshold)

    encoded_charts = list(map(encode_chart,
                              pltx,
                              [x_label] * len(pltx),
                              [y_label] * len(pltx),
                              [x_lim] * len(pltx),
                              [y_lim] * len(pltx),
                              [c_label] * len(pltx),
                              [color_scheme] * len(pltx),
                              [title] * len(pltx)
                              ))

    chart = reduce(lambda x, y: x + y, encoded_charts)
    return chart
