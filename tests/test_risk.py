import numpy as np
import pandas as pd

from quantdash.risk import historical_cvar, historical_var, max_drawdown


def test_var_and_cvar_sign_and_order():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.02] * 30)
    var = historical_var(returns)
    cvar = historical_cvar(returns)
    assert cvar <= var < 0


def test_max_drawdown():
    assert np.isclose(max_drawdown(pd.Series([0.10, -0.20, 0.05])), -0.20)
