# from src.env.robotics.fetch_env import FetchEnv
# from src.env.robotics.fetch_push import FetchPushEnv
def get_env(name):
    if name == "FetchPush":
        from src.env.robotics.fetch_push import FetchPushEnv

        return FetchPushEnv
