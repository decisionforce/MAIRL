from typing import Optional

from gym.envs import register as gym_register

_ENTRY_POINT_PREFIX = "envs.examples.airl_envs"


def _register(env_name: str, entry_point: str, max_episode_steps: Optional[int] = None, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, max_episode_steps=max_episode_steps, kwargs=kwargs)


def _point_maze_register():
    for dname, dval in {"Left": 0, "Right": 1}.items():
        for vname, vval in {"": False, "Vel": True}.items():
            _register(
                f"PointMaze{dname}{vname}-v0",
                entry_point="point_maze_env:PointMazeEnv",
                kwargs={"direction": dval, "include_vel": vval},
            )


_register(
    "ObjPusher-v0",
    entry_point=f"pusher_env:PusherEnv",
    kwargs={"sparse_reward": False},
)
_register("TwoDMaze-v0", entry_point="twod_maze:TwoDMaze")

_point_maze_register()

# A modified ant which flips over less and learns faster via TRPO
_register(
    "CustomAnt-v0",
    entry_point="ant_env:CustomAntEnv",
    max_episode_steps=1000,
    kwargs={"gear": 30, "disabled": False},
)
_register(
    "DisabledAnt-v0",
    entry_point="ant_env:CustomAntEnv",
    max_episode_steps=1000,
    kwargs={"gear": 30, "disabled": True},
)
