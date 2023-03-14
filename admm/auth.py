import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OUTPUTS'] = f"{ROOT}/outputs"
os.environ['BUS_CONFIG'] = f"{ROOT}/config/system_message_bus.yml"
os.environ['GOSS_CONFIG'] = f"{ROOT}/config/pnnl.goss.gridappsd.cfg"
os.environ['SIM_CONFIG'] = f"{ROOT}/config/ieee123.json"
os.environ['GRIDAPPSD_APPLICATION_ID'] = 'dist-sample-app'
os.environ['GRIDAPPSD_USER'] = 'system'
os.environ['GRIDAPPSD_PASSWORD'] = 'manager'
os.environ['GRIDAPPSD_ADDRESS'] = 'gridappsd'
os.environ['GRIDAPPSD_PORT'] = '61613'