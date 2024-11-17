from dotenv import load_dotenv
from centralized_nlp_package.utils import get_config


# load_dotenv()
# Initialize configuration first
config = get_config()

# Now set up logging using the initialized config
from centralized_nlp_package.utils import setup_logging


setup_logging()

# Expose submodules
from . import data_access
from . import data_processing
from . import common_utils
from . import configs
from . import embedding
from . import nli_utils
from . import text_processing


# Optionally, define __all__ to control what is exported
__all__ = [
    "data_access",
    "data_processing",
    "common_utils",
    "configs",
    "embedding",
    "nli_utils",
    "text_processing",
    "utils",
    # Add other modules or attributes you want to expose
]