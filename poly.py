"""
    This program looks at the feasibility of launching rockets into space using centripetal motion
    in the form of an extended arm that swings the rocket into orbit.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def change_in_velocity(initial: float, final: float, time: float):
    """
    Calculates the change in velocity of the rocket.

    :param initial: the initial velocity.
    :param final: the final velocity.
    :param time: the change in time (supposing initial_time = 0s).
    :return: the change in velocity.
    """
    return (final - initial) / time


def centripetal_acceleration(velocity: float, radius: float):
    """
    Calculates the centripetal acceleration of the rocket.

    :param velocity: the change in velocity of the rocket.
    :param radius: the length of the arm holding the rocket.
    :return: the acceleration of the rocket.
    """
    return velocity ** 2 / radius


"""
The velocity required to escape earth (Giga-metres/second). 
"""
escape_velocity: float = 11.186
"""
The amount of time the launch should take. 
"""
launch_time = 6
"""
The amount of samples to use for testing.
"""
samples = 100
randomness = 5
# Training set -
x_train = np.array([[velocity] for velocity in range(0, samples, 2)])
y_train = np.array(
    [centripetal_acceleration(change_in_velocity(escape_velocity, v_i, launch_time), 10) for v_i in x_train])
# Testing set, adding some random inverse velocity.
x_test = np.array([[v] for v in range(0, samples, 2)])
y_test = np.array([
    centripetal_acceleration(
        change_in_velocity(
            escape_velocity, initial_velocity, launch_time) - random.random() * randomness, 10) for
    index, initial_velocity in enumerate(x_test)])

regressor = LinearRegression()
regressor.fit(x_test, y_test)

x_regression_line = np.linspace(min(x_test), max(x_test), len(x_test))
y_regression_line = regressor.predict(x_regression_line.reshape(x_regression_line.shape[0], 1))

plt.plot(x_regression_line, y_regression_line)

quadratic_featurizer = PolynomialFeatures(degree=2)

x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_test_quadratic, y_test)
predicted_quadratic_output = quadratic_featurizer.transform(x_regression_line.reshape(x_regression_line.shape[0], 1))
# Plot the regression line
plt.plot(x_regression_line, regressor_quadratic.predict(predicted_quadratic_output), c='r', linestyle='--')
plt.title("Centripetal acceleration regressed on radius of motion")
plt.xlabel("Radius of motion(m)")
plt.ylabel("Centripetal acceleration x10⁶m/s²")
plt.axis([min(x_test) - 1, max(x_test) + 1, min(y_test) - 1, max(y_test) + 1])
plt.grid(True)
plt.scatter(x_test, y_test, s=10)
plt.show()
