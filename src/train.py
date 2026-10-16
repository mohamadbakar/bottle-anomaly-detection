from project import build_components, load_config


def main():
    try:
        config = load_config()
        datamodule, model, engine = build_components(config)
        engine.fit(model=model, datamodule=datamodule)
    except ValueError as exc:
        if "Embedding store is empty" in str(exc):
            raise RuntimeError(
                "PatchCore collected no training embeddings. Check the MVTecAD "
                "root/category and that bottle/train/good contains readable images."
            ) from exc
        raise


if __name__ == "__main__":
    main()
