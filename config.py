import yaml

global API_KEY
global CONFIG

with open("api.yml", "r") as api_file:
    API_KEY = yaml.safe_load(api_file).get("API_KEY")

with open("config.yml", "r") as config_file:
    CONFIG = yaml.safe_load(config_file)