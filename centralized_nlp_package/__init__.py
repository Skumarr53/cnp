from centralized_nlp_package.utils.config import get_config

# Initialize configuration first
config = get_config()

# Now set up logging using the initialized config
from centralized_nlp_package.utils.logging_setup import setup_logging
