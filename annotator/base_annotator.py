import os

# # default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
model_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "model"),
    )
)


class BaseProcessor:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device") if kwargs.get("device") is not None else "cpu"
        self.models_path = kwargs.get("models_path") if kwargs.get("models_path") is not None else model_cache_home

    def unload_model(self):
        pass
