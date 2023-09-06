import yaml

with open("models/cfg/FlowNetS.yaml", "r") as config:
    try:
        data = yaml.load(config, Loader=yaml.FullLoader)
        # print(data["layers"])
        for layer in data["layers"]:
            print(layer["name"], layer["out_channels"])
    except yaml.YAMLError as exc:
        print(exc)