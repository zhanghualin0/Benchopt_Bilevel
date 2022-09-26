from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state
    oracles = import_ctx.import_from('oracles')


class Objective(BaseObjective):
    name = "Bilevel Optimization"

    parameters = {
        'task, model, n_reg, reg': [
            ('datacleaning', None, None, None),
            ('classif', 'logreg', 'full', 'exp'),
            ('classif', 'logreg', 'full', 'lin'),
            ('classif', 'logreg', 1, 'exp'),
            ('classif', 'logreg', 1, 'lin'),
            ('classif', 'multilogreg', None, None),
            ('classif', 'ridge', 'full', 'exp'),
            ('classif', 'ridge', 'full', 'lin'),
            ('classif', 'ridge', 1, 'exp'),
            ('classif', 'ridge', 1, 'lin'),
        ],
    }

    def __init__(self, task='classif', model='ridge', reg='exp',  n_reg='full',
                 random_state=2442):
        if task == 'classif':
            self.reg = reg
            self.n_reg = n_reg
            if model == 'ridge':
                self.inner_oracle = oracles.RidgeRegressionOracle
                self.outer_oracle = oracles.RidgeRegressionOracle
            elif model == 'logreg':
                self.inner_oracle = oracles.LogisticRegressionOracle
                self.outer_oracle = oracles.LogisticRegressionOracle
            elif model == 'multilogreg':
                self.get_inner_oracle = oracles.MultiLogRegOracle
                self.get_outer_oracle = (
                    lambda X, y: oracles.MultiLogRegOracle(X, y, reg='none')
                )
            else:
                raise ValueError(
                    f"model should be 'ridge' or 'logreg'. Got '{model}'."
                )
        elif task == 'datacleaning':
            self.reg = 2e-1
            self.inner_oracle = oracles.DataCleaningOracle
            self.outer_oracle = oracles.MultinomialLogRegOracle
        else:
            raise ValueError(
                f"task should be 'classif' or 'datacleaning'. Got '{task}'"
            )
        self.task = task
        self.model = model
        self.random_state = random_state

    def get_one_solution(self):
        inner_shape, outer_shape = self.f_train.variables_shape
        return np.zeros(*inner_shape), np.zeros(*outer_shape)

    def set_data(self, X_train, y_train, X_test, y_test,
                 X_val=None, y_val=None):
        self.f_train = self.inner_oracle(
            X_train, y_train, reg=self.reg
        )
        if self.task == 'datacleaning':
            self.f_test = self.outer_oracle(
                X_test, y_test, reg=0.
            )
            self.X_val, self.y_val = X_val, y_val
        else:
            self.f_test = self.outer_oracle(
                X_test, y_test, reg='none'
            )

        rng = check_random_state(self.random_state)
        inner_shape, outer_shape = self.f_train.variables_shape
        self.inner_var0 = rng.randn(*inner_shape)
        if self.task == "classif":
            self.outer_var0 = rng.rand(*outer_shape)
        elif self.task == "datacleaning":
            self.outer_var0 = np.ones(*outer_shape)
        if self.reg == 'exp':
            self.outer_var0 = np.log(self.outer_var0)
        if self.n_reg == 1:
            self.outer_var0 = self.outer_var0[:1]
        self.inner_var0, self.outer_var0 = self.f_train.prox(
            self.inner_var0, self.outer_var0
        )

    def compute(self, beta):

        inner_var, outer_var = beta

        if np.isnan(outer_var).any():
            raise ValueError

        if self.task == 'classif' and self.model == 'logreg':
            inner_star = self.f_train.get_inner_var_star(outer_var)
            value_function = self.f_test.get_value(inner_star, outer_var)
            inner_value = self.f_train.get_value(inner_var, outer_var)
            outer_value = self.f_test.get_value(inner_var, outer_var)
            d_inner = np.linalg.norm(inner_var - inner_star)
            d_value = outer_value - value_function
            grad_f_test_inner, grad_f_test_outer = self.f_test.get_grad(
                inner_star, outer_var
            )
            grad_value = grad_f_test_outer
            v = self.f_train.get_inverse_hvp(
                inner_star, outer_var,
                grad_f_test_inner
            )
            grad_value -= self.f_train.get_cross(inner_star, outer_var, v)
            grad_inner = self.f_train.get_grad_inner_var(inner_var, outer_var)
            grad_star = self.f_train.get_grad_inner_var(inner_star, outer_var)

            return dict(
                value_func=value_function,
                inner_value=inner_value,
                outer_value=outer_value,
                d_inner=d_inner,
                d_value=d_value,
                value=np.linalg.norm(grad_value)**2,
                grad_inner=np.linalg.norm(grad_inner),
                grad_star=np.linalg.norm(grad_star),
            )
        elif self.task == 'datacleaning' or self.model == 'multilogreg':
            acc = self.f_test.accuracy(
                inner_var, outer_var, self.X_val, self.y_val
            )
            return dict(
                value=acc
            )

    def to_dict(self):
        return dict(
            f_train=self.f_train,
            f_test=self.f_test,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0
        )
