"""``python -m llm_memory_calculator.runtime.schema``: write runtime/schema/v1.json and runtime/settings/v1.json."""

from .generate import write_all

if __name__ == "__main__":
    for path in write_all():
        print(f"wrote {path}")
