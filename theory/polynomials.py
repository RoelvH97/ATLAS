"""Table 1: content-Jacobian geometry after three latent updates."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from data.coordinates import grid
from general_utils.checkpoints import sha256
from meta_reconstruction import ReconstructionTrainer


class PolynomialDiagnostics:
    def __init__(self, model, params, adaptation, chunk_size=64):
        self.model, self.params = model, params
        self.adaptation, self.chunk_size = adaptation, chunk_size
        self._directions = jax.jit(self.directions)

    def directions(self, coords, latent, vectors):
        structured = isinstance(latent, dict)
        content = latent["ctx"] if structured else latent

        def decode(vector):
            code = (
                {**latent, "ctx": vector.reshape(content.shape)}
                if structured
                else vector.reshape(content.shape)
            )
            return self.model.apply({"params": self.params["model"]}, coords, code)

        return jax.vmap(lambda vector: jax.jvp(decode, (content.ravel(),), (vector,))[1])(vectors)

    def measure(self, coords, target, support, basis):
        latent = self.adaptation.fit(self.params, self.model, coords[support], target[support])
        structured = isinstance(latent, dict)
        content = latent["ctx"] if structured else latent
        rates = self.params.get("meta_lr", self.adaptation.rates(latent))
        rates = rates["ctx"] if structured else rates
        metric = np.broadcast_to(np.asarray(rates), content.shape).ravel().astype(np.float64)
        if not np.all(metric > 0):
            raise ValueError("Geometry requires a positive scalar or shared content step size")

        identity = np.eye(content.size, dtype=np.float32)
        columns = [
            np.asarray(
                self._directions(coords, latent, jnp.asarray(identity[i : i + self.chunk_size]))
            )
            .reshape(min(self.chunk_size, content.size - i), -1)
            .T
            for i in range(0, content.size, self.chunk_size)
        ]
        weighted = np.concatenate(columns, axis=1).astype(np.float64) * np.sqrt(metric)[None]
        eigenvalues = np.linalg.eigvalsh(weighted @ weighted.T)
        positive = eigenvalues[eigenvalues > 1e-7 * max(float(eigenvalues[-1]), 1e-300)]
        probability = positive / positive.sum() if len(positive) else positive
        effective_rank = (
            float(np.exp(-np.sum(probability * np.log(probability)))) if len(positive) else 0.0
        )
        orthogonal, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64), mode="reduced")
        tangent_fraction = float(
            np.sum((orthogonal.T @ weighted) ** 2) / (np.sum(weighted**2) + 1e-300)
        )
        prediction = np.asarray(self.model.apply({"params": self.params["model"]}, coords, latent))
        mse = float(np.mean((prediction - np.asarray(target)) ** 2))
        return {
            "mse": mse,
            "psnr": float(10 * np.log10(4 / max(mse, 1e-300))),
            "effective_rank": effective_rank,
            "tangent_fraction": tangent_fraction,
        }


class PolynomialExperiment(ReconstructionTrainer):
    def run(self):
        if self.config["mode"] != "analyze":
            return super().run()
        if not self.config["checkpoint"]:
            raise ValueError("Analysis requires checkpoint=PATH")
        cfg = self.config["polynomial_evaluation"]
        dataset = self.data.dataset("val").dataset
        coords = grid(dataset.size, array_module=jnp)
        basis = dataset.basis(np.asarray(coords))
        rng = np.random.default_rng(cfg["fit_seed"])
        diagnostics = PolynomialDiagnostics(
            self.model, self.params, self.adaptation, cfg["jacobian_chunk"]
        )
        rows = []
        for index in range(cfg["fields"]):
            values = jnp.asarray(dataset[index][0].reshape(-1, 1))
            support = np.sort(
                rng.choice(len(coords), self.config["data"]["inner_coords"], replace=False)
            )
            rows.append(diagnostics.measure(coords, values, support, basis))
            print(json.dumps({"field": index, **rows[-1]}), flush=True)
        result = {
            "seed": self.training["seed"],
            "model": self.config["model"],
            "gradient_rule": self.adaptation.gradient_rule,
            "checkpoint_sha256": sha256(self.config["checkpoint"]),
            "fields": rows,
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "protocol": {
                "pool": "validation",
                "fit_seed": cfg["fit_seed"],
                "support": self.config["data"]["inner_coords"],
                "queries": len(coords),
                "steps": self.adaptation.steps,
            },
        }
        (self.output / "polynomial_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
