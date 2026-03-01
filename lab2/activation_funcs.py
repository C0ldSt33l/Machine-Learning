from math import exp

def logistic(x: float) -> float:
    return 1 / (1 + exp(-x))

def logistic_derivate(x: float) -> float:
    log_calc = logistic(x)
    return log_calc * (1 - log_calc)


def hyperbolic_tangent(x: float) -> float:
    e_in_x = exp(x)
    e_in_neg_x = exp(-x)
    return (e_in_x - e_in_neg_x) / (e_in_x + e_in_neg_x)

def hyperbolic_tangent_derivate(x: float) -> float:
    return 1 - hyperbolic_tangent(x)


# TODO: search about rational sigmioda & its derivate
def soft_sign(x: float) -> float:
    return 1 / (1 + abs(x))

def soft_sign_derivate(x: float) -> float:
    return 1 / ((1 + abs(x)) ** 2)


def relu(x: float) -> float:
    return 0 if x < 0 else x

def relu_derivate(x: float) -> float:
    return 0 if x < 0 else 1


leaky_relu_mult: float = 0.1
def leaky_relu(x: float) -> float:
    return leaky_relu_mult * x if x < 0 else x

def leaky_relu(x: float) -> float:
    return leaky_relu_mult if x < 0 else x