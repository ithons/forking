y(t_s[i], t_p[i], w_m, w_p, estimator, t_min, t_max) for i in range(len(t_s))]
    print(-np.sum(np.log(likelihoods)))
    return -np.sum(np.log(likelihoo