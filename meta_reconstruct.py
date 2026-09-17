import hydra

from meta_reconstruction import ReconstructionTrainer


@hydra.main(version_base=None, config_path="configs", config_name="reconstruct")
def main(config):
    ReconstructionTrainer(config).run()


if __name__ == "__main__":
    main()
