import hydra

from meta_segmentation import SegmentationTrainer


@hydra.main(version_base=None, config_path="configs", config_name="segment")
def main(config):
    SegmentationTrainer(config).run()


if __name__ == "__main__":
    main()
