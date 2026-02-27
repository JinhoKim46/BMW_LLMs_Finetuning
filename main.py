from src.bmw_master import Master

CONFIG_PATH = "config/config.yaml"

if __name__ == "__main__":
    master = Master(CONFIG_PATH)
    master.run()
