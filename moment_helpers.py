from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from DTMCMC.likelihood import AbstractLikelihood


def get_averaged_means[LikelihoodType: AbstractLikelihood[Any]](
    mcc: DTMCMCSampler[LikelihoodType], length: int, cut: int = 0
) -> list[NDArray[np.floating]]:
    l1_means = np.array(mcc.logL_means)
    l2_means = np.array(mcc.logL2_means)
    l3_means = np.array(mcc.logL3_means)
    l4_means = np.array(mcc.logL4_means)
    l5_means = np.array(mcc.logL5_means)
    l6_means = np.array(mcc.logL6_means)

    l1_avg = np.mean(l1_means[cut:].reshape(((l1_means.shape[0] - cut) // length, length, l1_means.shape[1])), axis=1)
    l2_avg = np.mean(l2_means[cut:].reshape(((l2_means.shape[0] - cut) // length, length, l2_means.shape[1])), axis=1)
    l3_avg = np.mean(l3_means[cut:].reshape(((l3_means.shape[0] - cut) // length, length, l3_means.shape[1])), axis=1)
    l4_avg = np.mean(l4_means[cut:].reshape(((l4_means.shape[0] - cut) // length, length, l4_means.shape[1])), axis=1)
    l5_avg = np.mean(l5_means[cut:].reshape(((l5_means.shape[0] - cut) // length, length, l5_means.shape[1])), axis=1)
    l6_avg = np.mean(l6_means[cut:].reshape(((l6_means.shape[0] - cut) // length, length, l6_means.shape[1])), axis=1)
    return [l1_avg, l2_avg, l3_avg, l4_avg, l5_avg, l6_avg]


def get_cumulants(l_means: list[NDArray[np.floating]]) -> list[NDArray[np.floating]]:
    l1_means = l_means[0]  # np.array(mcc.logL_means)
    l2_means = l_means[1]  # np.array(mcc.logL2_means)
    l3_means = l_means[2]  # np.array(mcc.logL3_means)
    l4_means = l_means[3]  # np.array(mcc.logL4_means)
    l5_means = l_means[4]  # np.array(mcc.logL4_means)
    l6_means = l_means[5]  # np.array(mcc.logL4_means)

    # l_vars = l2_means - l1_means**2

    l_cum1 = l1_means
    l_cum2 = l2_means - l1_means**2
    l_cum3 = l3_means - 3 * l1_means * l2_means + 2 * l1_means**3
    l_cum4 = l4_means - 4 * l3_means * l1_means - 3 * l2_means**2 + 12 * l2_means * l1_means**2 - 6 * l1_means**4
    l_cum5 = (
        l5_means
        - 5 * l4_means * l1_means
        - 10 * l3_means * l2_means
        + 20 * l3_means * l1_means**2
        + 30 * l2_means**2 * l1_means
        - 60 * l2_means * l1_means**3
        + 24 * l1_means**5
    )
    l_cum6 = (
        l6_means
        - 6 * l5_means * l1_means
        - 15 * l4_means * l2_means
        + 30 * l4_means * l1_means**2
        - 10 * l3_means**2
        + 120 * l3_means * l2_means * l1_means
        - 120 * l3_means * l1_means**3
        + 30 * l2_means**3
        - 270 * l2_means**2 * l1_means**2
        + 360 * l2_means * l1_means**4
        - 120 * l1_means**6
    )

    # l_skews = l_cum3/l_vars**(3/2)
    return [l_cum1, l_cum2, l_cum3, l_cum4, l_cum5, l_cum6]


def get_averaged_adjacents[LikelihoodType: AbstractLikelihood[Any]](
    mcc: DTMCMCSampler[LikelihoodType], length: int, cut: int = 0
) -> list[NDArray[np.floating]]:
    l_p11_means = np.array(mcc.logL_prod11_means)
    l_p21_means = np.array(mcc.logL_prod21_means)
    l_p12_means = np.array(mcc.logL_prod12_means)

    l_p11_avg: NDArray[np.floating] = np.mean(
        l_p11_means[cut:].reshape(((l_p11_means.shape[0] - cut) // length, length, l_p11_means.shape[1])), axis=1
    )
    l_p21_avg: NDArray[np.floating] = np.mean(
        l_p21_means[cut:].reshape(((l_p11_means.shape[0] - cut) // length, length, l_p21_means.shape[1])), axis=1
    )
    l_p12_avg: NDArray[np.floating] = np.mean(
        l_p12_means[cut:].reshape(((l_p11_means.shape[0] - cut) // length, length, l_p12_means.shape[1])), axis=1
    )

    return [l_p11_avg, l_p21_avg, l_p12_avg]


def get_corr_quantities(
    l_means: list[NDArray[np.floating]], l_adjacents: list[NDArray[np.floating]]
) -> tuple[list[NDArray[np.floating]], list[NDArray[np.floating]]]:
    l_p11 = l_adjacents[0]
    l_p21 = l_adjacents[1]
    l_p12 = l_adjacents[2]

    n_chain = l_means[0].shape[-1]

    x_mean = l_means[0][:, : n_chain - 1]
    y_mean = l_means[0][:, 1:]

    x2_mean = l_means[1][:, : n_chain - 1]
    y2_mean = l_means[1][:, 1:]

    x_std = np.sqrt(x2_mean - x_mean**2)
    y_std = np.sqrt(y2_mean - y_mean**2)

    l_cov = l_adjacents[0] - x_mean * y_mean

    l_coskew21 = (
        l_p21 - 2 * x_mean * l_p11 + x_mean**2 * y_mean - x2_mean * y_mean + 2 * x_mean**2 * y_mean - x_mean**2 * y_mean
    )
    l_coskew12 = (
        l_p12 - 2 * y_mean * l_p11 + y_mean**2 * x_mean - y2_mean * x_mean + 2 * y_mean**2 * x_mean - y_mean**2 * x_mean
    )

    return [l_cov, l_coskew12, l_coskew21], [
        l_cov / (x_std * y_std),
        l_coskew21 / (x_std**2 * y_std),
        l_coskew12 / (x_std * y_std**2),
    ]
