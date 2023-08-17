from src.common.registry import registry
from src.utils.utils import setup_imports

setup_imports()

entity= 'callback_name_mapping'
print(registry.mapping[entity])