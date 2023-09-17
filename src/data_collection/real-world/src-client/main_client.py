from vcqoe import cli
import toml

""" vcqoe command-line interface entry-point """
config_file = "config.toml"
#cli.execute(config_file)
cli.execute_server(config_file)
