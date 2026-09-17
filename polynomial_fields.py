import hydra

from theory.polynomials import PolynomialExperiment


@hydra.main(version_base=None, config_path="configs", config_name="polynomials")
def main(config):
    PolynomialExperiment(config).run()


if __name__ == "__main__":
    main()
