"""Pydantic compatibility layer to support both v1 and v2."""

from typing import Type

# Try to import Pydantic v2 first
try:
    from pydantic import BaseModel as BaseModelV2
    from pydantic import Field as FieldV2
    from pydantic import field_validator, model_validator

    PYDANTIC_V2 = True
    PYDANTIC_VERSION = 2

    # Create v1-compatible decorators for v2
    def validator(*fields, **kwargs):
        """Compatibility wrapper for Pydantic v2 field_validator."""

        def decorator(func):
            # Extract common kwargs
            mode = kwargs.get("mode", "after")
            check_fields = kwargs.get("check_fields", True)

            # For v2, we need to apply field_validator to each field
            if fields:
                return field_validator(*fields, mode=mode, check_fields=check_fields)(func)
            return func

        return decorator

    def root_validator(pre: bool = False, **kwargs):
        """Compatibility wrapper for Pydantic v2 model_validator."""
        mode = "before" if pre else "after"

        def decorator(func):
            return model_validator(mode=mode)(func)

        return decorator

    # Use v2 exports
    BaseModel = BaseModelV2
    Field = FieldV2

except ImportError:
    # Fall back to Pydantic v1
    try:
        from pydantic import BaseModel as BaseModelV1
        from pydantic import Field as FieldV1
        from pydantic import root_validator as root_validator_v1
        from pydantic import validator as validator_v1

        PYDANTIC_V2 = False
        PYDANTIC_VERSION = 1

        # Use v1 exports directly
        BaseModel = BaseModelV1
        Field = FieldV1
        validator = validator_v1
        root_validator = root_validator_v1

    except ImportError:
        # No Pydantic available
        PYDANTIC_V2 = False
        PYDANTIC_VERSION = 0

        # Create dummy classes for when Pydantic is not available
        class BaseModel:
            """Dummy BaseModel when Pydantic is not available."""

            def __init__(self, **data):
                for key, value in data.items():
                    setattr(self, key, value)

            def dict(self):
                """Get dictionary representation."""
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            def json(self):
                """Get JSON representation."""
                import json

                return json.dumps(self.dict())

            @classmethod
            def parse_obj(cls, obj):
                """Parse object."""
                return cls(**obj)

            @classmethod
            def parse_file(cls, path):
                """Parse file."""
                import json

                with open(path) as f:
                    return cls(**json.load(f))

        def Field(default=None, **kwargs):
            """Dummy Field function."""
            return default

        def validator(*fields, **kwargs):
            """Dummy validator decorator."""

            def decorator(func):
                return func

            return decorator

        def root_validator(**kwargs):
            """Dummy root_validator decorator."""

            def decorator(func):
                return func

            return decorator


# Helper functions for version-specific behavior
def get_pydantic_version() -> int:
    """Get the major version of Pydantic (0 if not installed)."""
    return PYDANTIC_VERSION


def is_pydantic_available() -> bool:
    """Check if any version of Pydantic is available."""
    return PYDANTIC_VERSION > 0


def create_model_config(**kwargs) -> Type:
    """Create a model config class compatible with the installed Pydantic version."""
    if PYDANTIC_V2:
        # For Pydantic v2, we use ConfigDict
        from pydantic import ConfigDict

        return ConfigDict(**kwargs)
    elif PYDANTIC_VERSION == 1:
        # For Pydantic v1, we create a Config class
        class Config:
            pass

        for key, value in kwargs.items():
            setattr(Config, key, value)
        return Config
    else:
        # No Pydantic, return empty class
        class Config:
            pass

        return Config


# Export all compatibility items
__all__ = [
    "BaseModel",
    "Field",
    "validator",
    "root_validator",
    "get_pydantic_version",
    "is_pydantic_available",
    "create_model_config",
    "PYDANTIC_V2",
    "PYDANTIC_VERSION",
]
