LIBRARY_REGISTRY = {}


def register_library(cls):
    """Decorator to register SPS library handlers."""
    if not hasattr(cls, "name"):
        raise ValueError("Library handler must define a 'name' attribute")
    LIBRARY_REGISTRY[cls.name.upper()] = cls
    return cls
