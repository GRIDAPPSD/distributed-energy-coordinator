import os
import auth
import admm

if __name__ == "__main__":
    print("goss config: ", os.environ.get('GOSS_CONFIG'))
    print("bus config: ", os.environ.get('BUS_CONFIG'))
    print("sim config: ", os.environ.get('SIM_CONFIG'))

    admm.run()