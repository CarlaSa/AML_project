import torch
import json


def main():
    if not torch.cuda.is_available():
        print({})
    else:
        props = torch.cuda.get_device_properties(0)
        print(json.dumps(dict(device_count=torch.cuda.device_count(),
                              **{key: getattr(props, key)
                                 for key in dir(props)
                                 if not key.startswith("__")})))


if __name__ == '__main__':
    main()
