import yaml

def parse_configs(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model = config_dict.get("model")
    sampler = config_dict.get("sampler")
    
    return model, sampler