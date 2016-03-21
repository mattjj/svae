from __future__ import division

from ..util import contract, sub
import niw, mniw


def lds_prior_expectedstats(natparam):
    niw_natparam, mniw_natparam = natparam
    return niw.expectedstats(niw_natparam), mniw.expectedstats(mniw_natparam)


def lds_prior_logZ(natparam):
    niw_natparam, mniw_natparam = natparam
    return niw.logZ(niw_natparam) + mniw.logZ(mniw_natparam)


def lds_prior_vlb(global_natparam, prior_natparam, expected_stats=None):
    if expected_stats is None:
        expected_stats = lds_prior_expectedstats(global_natparam)
    return contract(sub(prior_natparam, global_natparam), expected_stats) \
        - (lds_prior_logZ(prior_natparam) - lds_prior_logZ(global_natparam))
