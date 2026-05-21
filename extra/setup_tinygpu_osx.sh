#!/bin/sh
python3 -c "
try:
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    APLRemotePCIDevice.ensure_app()
except Exception as e:
    print('Your tinygrad-arkey is too old. Please clone the latest tinygrad-arkey: git clone https://github.com/JulianAbeleda/tinygrad-arkey.git && cd tinygrad-arkey && python3 -m pip install -e .')
    print(e)
    exit(1)
"
