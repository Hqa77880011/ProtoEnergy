import numpy as np
from scipy.special import logsumexp, softmax


def normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


class ProtoEnergy:
    def __init__(self, temperature=1, alpha=0.5):
        self.temperature, self.alpha = temperature, alpha

    def geometry(self, features, logits, radius=True):
        prediction = logits.argmax(1)
        distance = np.linalg.norm(normalize(features) - self.prototype[prediction], axis=1)
        if radius:
            distance /= self.radius[prediction] + 1e-6
        energy = -self.temperature * logsumexp(logits / self.temperature, axis=1)
        return prediction, energy, distance

    def fit(self, training_features, training_labels, validation_features, validation_logits):
        z = normalize(training_features)
        count = validation_logits.shape[1]
        self.prototype = np.stack([z[training_labels == k].sum(0) for k in range(count)])
        if any(not np.any(training_labels == k) for k in range(count)):
            raise ValueError("Every class needs ID training features")
        self.prototype = normalize(self.prototype)
        distances = np.linalg.norm(z - self.prototype[training_labels], axis=1)
        self.radius = np.array([np.quantile(distances[training_labels == k], 0.95) for k in range(count)])
        self.mean_radius = np.array([np.mean(distances[training_labels == k]) for k in range(count)])
        prediction, energy, deviation = self.geometry(validation_features, validation_logits)
        if any(not np.any(prediction == k) for k in range(count)):
            raise ValueError("Every predicted class needs ID calibration support")
        self.references = {k: (np.sort(energy[prediction == k]), np.sort(deviation[prediction == k])) for k in range(count)}
        raw_distance = self.geometry(validation_features, validation_logits, False)[2]
        self.raw_references = {k: np.sort(raw_distance[prediction == k]) for k in range(count)}
        self.pooled = (np.sort(energy), np.sort(deviation))
        self.validation_features, self.validation_logits = validation_features, validation_logits
        self.thresholds = {mode: float(np.quantile(self.score(validation_features, validation_logits, mode), 0.95)) for mode in ["full", "pooled", "no_radius", "energy", "prototype", "radius", "raw", "mean_radius", "quantile_radius", "msp", "maxlogit"]}
        return self

    def score(self, features, logits, mode="full"):
        predicted, energy, deviation = self.geometry(features, logits, mode not in {"prototype", "raw", "no_radius"})
        if mode == "energy":
            return energy
        if mode == "msp":
            return -softmax(logits, axis=1).max(1)
        if mode == "maxlogit":
            return -logits.max(1)
        if mode in {"prototype", "radius"}:
            return deviation
        if mode == "mean_radius":
            raw = self.geometry(features, logits, False)[2]
            return self.alpha * energy + (1 - self.alpha) * raw / (self.mean_radius[predicted] + 1e-6)
        if mode in {"raw", "quantile_radius"}:
            return self.alpha * energy + (1 - self.alpha) * deviation
        scores = np.empty(len(features))
        for k, (reference_energy, reference_distance) in self.references.items():
            mask = predicted == k
            if mode == "pooled":
                reference_energy, reference_distance = self.pooled
            if mode != "pooled":
                reference_distance = self.raw_references[k]
                query_distance = self.geometry(features[mask], logits[mask], False)[2]
            else:
                query_distance = deviation[mask]
            qe = np.searchsorted(reference_energy, energy[mask], side="right") / len(reference_energy)
            qd = np.searchsorted(reference_distance, query_distance, side="right") / len(reference_distance)
            scores[mask] = self.alpha * qe + (1 - self.alpha) * qd
        return scores

    def save(self, path):
        arrays = {"prototype": self.prototype, "radius": self.radius, "mean_radius": self.mean_radius,
                  "validation_features": self.validation_features, "validation_logits": self.validation_logits,
                  "temperature": self.temperature, "alpha": self.alpha}
        for k, (energy, distance) in self.references.items():
            arrays[f"energy_{k}"], arrays[f"distance_{k}"] = energy, distance
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        obj = cls(float(data["temperature"]), float(data["alpha"]))
        for key in ["prototype", "radius", "mean_radius", "validation_features", "validation_logits"]:
            setattr(obj, key, data[key])
        obj.references = {k: (data[f"energy_{k}"], data[f"distance_{k}"]) for k in range(len(obj.radius))}
        predicted, _, raw = obj.geometry(obj.validation_features, obj.validation_logits, False)
        obj.raw_references = {k: np.sort(raw[predicted == k]) for k in range(len(obj.radius))}
        obj.pooled = tuple(np.sort(np.concatenate([obj.references[k][j] for k in obj.references])) for j in range(2))
        obj.thresholds = {mode: float(np.quantile(obj.score(obj.validation_features, obj.validation_logits, mode), 0.95)) for mode in ["full", "pooled", "no_radius", "energy", "prototype", "radius", "raw", "mean_radius", "quantile_radius", "msp", "maxlogit"]}
        return obj
