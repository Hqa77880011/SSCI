from itertools import combinations_with_replacement, product

import numpy as np


def quantile(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= len(scores) else np.inf


class ForwardModel:
    def fit(self, states, context, responses, development, ridge=1e-5):
        inputs = np.c_[states, context]
        self.center, self.scale = inputs.mean(0), inputs.std(0).clip(1e-8)
        self.pairs = list(combinations_with_replacement(range(inputs.shape[1]), 2))
        design = self.features(inputs)
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0
        self.coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ responses)
        dev_states, dev_context, dev_responses = development
        residual = dev_responses - self.predict(dev_states, dev_context)
        self.coefficients[0] += residual.mean(0)
        self.response_scale = (residual - residual.mean(0)).std(0, ddof=1).clip(1e-6)
        self.state_dimensions = states.shape[1]
        return self

    def features(self, inputs):
        normalized = (inputs - self.center) / self.scale
        terms = [np.ones((len(inputs), 1)), normalized]
        terms += [(normalized[:, i] * normalized[:, j])[:, None] for i, j in self.pairs]
        return np.concatenate(terms, axis=1)

    def predict(self, states, context):
        return self.features(np.c_[states, context]) @ self.coefficients

    def bounds(self, lows, highs, context):
        count = len(lows)
        lo = np.c_[lows, np.repeat(context[None], count, axis=0)]
        hi = np.c_[highs, np.repeat(context[None], count, axis=0)]
        lo, hi = (lo - self.center) / self.scale, (hi - self.center) / self.scale
        lower, upper = [np.ones((count, 1)), lo], [np.ones((count, 1)), hi]
        for i, j in self.pairs:
            corners = np.stack([lo[:, i] * lo[:, j], lo[:, i] * hi[:, j],
                                hi[:, i] * lo[:, j], hi[:, i] * hi[:, j]])
            minimum, maximum = corners.min(0), corners.max(0)
            if i == j:
                minimum = np.where((lo[:, i] <= 0) & (hi[:, i] >= 0), 0, minimum)
            lower.append(minimum[:, None])
            upper.append(maximum[:, None])
        lower, upper = np.concatenate(lower, 1), np.concatenate(upper, 1)
        positive, negative = np.maximum(self.coefficients, 0), np.minimum(self.coefficients, 0)
        low = lower @ positive + upper @ negative
        high = upper @ positive + lower @ negative
        tolerance = 1e-10 * np.maximum(1, np.maximum(abs(low), abs(high)))
        return low - tolerance, high + tolerance

    def jacobian(self, state, context):
        inputs = (np.r_[state, context] - self.center) / self.scale
        gradient = self.coefficients[1:1 + len(inputs)].copy()
        for index, (i, j) in enumerate(self.pairs):
            coefficient = self.coefficients[1 + len(inputs) + index]
            gradient[i] += coefficient * inputs[j]
            gradient[j] += coefficient * inputs[i]
        return (gradient / self.scale[:, None])[:self.state_dimensions].T


def cells(domain, subdivisions):
    if len(domain) != len(subdivisions):
        raise ValueError("One subdivision count per state dimension required")
    if any(n < 1 for n in subdivisions):
        raise ValueError("Subdivision counts must be positive")
    axes = [np.linspace(low, high, n + 1) for (low, high), n in zip(domain, subdivisions)]
    indices = np.array(list(product(*(range(n) for n in subdivisions))))
    lows = np.stack([axis[indices[:, i]] for i, axis in enumerate(axes)], 1)
    highs = np.stack([axis[indices[:, i] + 1] for i, axis in enumerate(axes)], 1)
    return lows, highs


class ResponseService:
    def __init__(self, vector):
        self._vector = vector
        self.queried = []

    def query(self, channel):
        if channel in self.queried:
            raise ValueError("A channel may only be charged and revealed once")
        self.queried.append(int(channel))
        return float(self._vector[channel])


def components(lows, highs):
    order = np.argsort(lows)
    output = []
    for index in order:
        if output and lows[index] <= output[-1][1] + 1e-12:
            output[-1][1] = max(output[-1][1], float(highs[index]))
        else:
            output.append([float(lows[index]), float(highs[index])])
    return output


def invert(model, context, service, domain, subdivisions, q, budget=6,
           epsilon=.5, policy="separation", seed=0, variant="outer", costs=None):
    lows, highs = cells(domain, subdivisions)
    if variant == "plugin":
        nuisance = np.mean(np.asarray(domain), axis=1)
        lows[:, 1:], highs[:, 1:] = nuisance[1:], nuisance[1:]
    if variant == "point_grid":
        lows = (lows + highs) / 2
        highs = lows.copy()
    response_low, response_high = model.bounds(lows, highs, context)
    cache_bytes = int(lows.nbytes + highs.nbytes + response_low.nbytes + response_high.nbytes)
    channel_count = response_low.shape[1]
    costs = np.ones(channel_count) if costs is None else np.asarray(costs)
    if costs.shape != (channel_count,) or np.any(costs <= 0):
        raise ValueError("Readout costs must be positive, with one value per channel")
    keep = np.ones(len(lows), bool)
    rng = np.random.default_rng(seed)
    charged = 0.
    while keep.any():
        hull = [float(lows[keep, 0].min()), float(highs[keep, 0].max())]
        if hull[1] - hull[0] <= 2 * epsilon:
            break
        affordable = [j for j in range(channel_count) if j not in service.queried and costs[j] <= budget - charged + 1e-12]
        if not affordable:
            break
        if policy == "random":
            channel = int(rng.choice(affordable))
        elif policy == "fixed":
            channel = affordable[0]
        else:
            retained = np.flatnonzero(keep)
            first = rng.choice(retained, 2048, replace=True)
            second = rng.choice(retained, 2048, replace=True)
            concentration_gap = np.maximum(lows[first, 0] - highs[second, 0], lows[second, 0] - highs[first, 0])
            utilities = []
            for j in affordable:
                response_gap = np.maximum(response_low[first, j] - response_high[second, j],
                                          response_low[second, j] - response_high[first, j])
                utilities.append(np.sum((concentration_gap > 2 * epsilon) &
                                        (response_gap > 2 * q * model.response_scale[j])) / costs[j])
            channel = affordable[int(np.argmax(utilities))] if max(utilities) > 0 else min(affordable, key=lambda j: costs[j])
        observed = service.query(channel)
        distance = np.maximum(np.maximum(response_low[:, channel] - observed,
                                         observed - response_high[:, channel]), 0)
        keep &= distance <= q * model.response_scale[channel]
        charged += costs[channel]
    if not keep.any():
        return {"interval": None, "components": [], "point": None, "accepted": False,
                "incompatible": True, "charged_cost": charged, "channels": service.queried,
                "retained_cells": 0, "inversion_cache_bytes": cache_bytes}
    projected = components(lows[keep, 0], highs[keep, 0])
    hull = [projected[0][0], projected[-1][1]]
    midpoint = .5 * (hull[0] + hull[1])
    center = np.mean(.5 * (lows[keep] + highs[keep]), axis=0)
    chosen = service.queried
    gamma = 0.
    if chosen:
        jacobian = model.jacobian(center, context)[chosen] / model.response_scale[chosen, None]
        concentration, nuisance = jacobian[:, 0], jacobian[:, 1:]
        residual = concentration - nuisance @ np.linalg.lstsq(nuisance, concentration, rcond=None)[0]
        gamma = float(np.linalg.norm(residual))
    return {"interval": hull, "components": projected, "point": midpoint,
            "accepted": hull[1] - hull[0] <= 2 * epsilon, "incompatible": False,
            "charged_cost": charged, "channels": service.queried, "retained_cells": int(keep.sum()),
            "gamma": gamma, "inversion_cache_bytes": cache_bytes}
