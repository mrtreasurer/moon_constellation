#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:15:37 2019

@author: bert
"""

import numpy as np

from numba import jit

from definitions import legendre, cos_lambda, sin_lambda


@jit
def d2u_dx2(pos, r_c, mu, c, s, n_max, m_max):
    r = np.linalg.norm(pos)

    sin_phi = pos[2]/r

    rc_r = r_c/r
    rc_rn = r_c/r

    l_sum = 2
    m_sum = 0
    n_sum = 0
    o_sum = 0
    p_sum = 0
    q_sum = 0
    r_sum = 0
    s_sum = 0
    t_sum = 0
    g_sum = 1
    h_sum = 0

    if n_max >= 2:
        cos_l = cos_lambda(pos, n_max)
        sin_l = sin_lambda(pos, n_max)

        p = legendre(n_max + 2, sin_phi)

        cm = np.zeros(m_max + 1)
        sm = np.zeros(m_max + 1)

        for n in range(2, m_max + 1):
            rc_rn *= rc_r

            g_sum_n = c[n, 0] * (n + 1) * p[n, 0]
            h_sum_n = c[n, 0] * p[n, 1]

            for m in range(0, n + 1):
                rho_rm = (pos[0]/r)**m + (pos[1]/r)**m

                cm[m] = rho_rm * cos_l[m]
                sm[m] = rho_rm * sin_l[m]

                bnm = c[n, m] * cm[m] + s[n, m] * sm[m]

                l_sum += rc_rn * (n + m + 1) * (n + m + 2) * p[n, m] * bnm

                if m == 0:
                    norm = np.sqrt(n * (n - 1) * (n + 1) * (n + 2) / 2)

                elif m == 1:
                    norm = np.sqrt((n - 1) * (n - 2) * (n + 2) * (n + 3))

                else:
                    norm = np.sqrt((n - m) * (n - m - 1) * (n + m + 1) * (n + m + 2))

                m_sum += rc_rn * norm * p[n, m + 2] * bnm

                if m >= 2:
                    n_sum += rc_rn * p[n, m] * m * (m - 1) * \
                        (c[n, m] * cm[m - 2] + s[n, m] * sm[m - 2])

                    o_sum += rc_rn * p[n, m] * m * (m - 1) * \
                        (c[n, m] * sm[m - 2] - s[n, m] * cm[m - 2])

                p_sum += rc_rn * p[n, m + 1] * (m + n + 1) * bnm

                if m >= 1:
                    q_sum += rc_rn * p[n, m + 1] * \
                        (c[n, m] * cm[m - 1] + s[n, m] * sm[m - 1])

                    r_sum -= rc_rn * p[n, m + 1] * \
                        (c[n, m] * sm[m - 1] - s[n, m] * cm[m - 1])

                    s_sum += rc_rn * (m + n + 1) * p[n, m] * m * \
                        (c[n, m] * sm[m - 1] - s[n, m] * cm[m - 1])

                    t_sum -= rc_rn * (m + n + 1) * p[n, m] * m * \
                        (c[n, m] * sm[m - 1] - s[n, m] * cm[m - 1])

                    g_sum_n += (1 + n + m) * p[n, m] * (c[n, m] * cm[m] + s[n, m] * sm[m])

                    h_sum_n += np.sqrt((n - m) * (n + m + 1)) * p[n, m + 1] * \
                        (c[n, m] * cm[m] + s[n, m] * sm[m])

            g_sum += rc_rn * g_sum_n
            h_sum += rc_rn * h_sum_n

    pos_r = pos/r
    alpha = np.array([q_sum, r_sum, 0])
    lamb = (g_sum + sin_phi * h_sum)
    f = l_sum + sin_phi * (m_sum * sin_phi + 2*(p_sum + h_sum)) + lamb
    g = -(m_sum*sin_phi + p_sum + h_sum)
    y = np.array([s_sum, t_sum, 0])
    d = sin_phi*alpha + y

    ddu = mu/r**3 * (np.dot(np.dot(np.column_stack((pos_r, alpha)),
                                   np.array([[f, g], [g, m_sum]])),
                            np.row_stack((pos_r, alpha)))
                     + np.dot(np.dot(np.column_stack((pos_r, d)),
                                     np.array([[0., -1.], [-1., 0.]])),
                              np.row_stack((pos_r, d)))
                     + np.array([[n_sum - lamb, -o_sum, q_sum],
                                 [-o_sum, -(n_sum + lamb), r_sum],
                                 [q_sum, r_sum, -lamb]]))

    return ddu


if __name__ == "__main__":
    import constants as cte

    x0 = -1.311519120505e-2
    y0 = 5.435394815081e-4
    z0 = 0.

    pos = np.array([x0, y0, z0])

    print(d2u_dx2(pos, cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order))

    for n in range(cte.degree):
        dacc = d2u_dx2(pos, cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.degree)