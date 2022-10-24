import argparse
from argparse import ArgumentParser


def str2bool(v):
    return v.lower() == "true"


def str2intlist(value):
    if not value:
        return value
    else:
        return [int(num) for num in value.split(",")]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(",")]


def create_parser():
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        "Robot Aware Cost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jobname", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default="pal")
    parser.add_argument("--wandb_project", type=str, default="roboaware")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default=None)

    parser.add_argument("--visualize_mode", type=str2bool, default=False)

    add_method_arguments(parser)
    add_ensemble_arguments(parser)

    return parser


def add_method_arguments(parser: ArgumentParser):
    # method arguments
    parser.add_argument(
        "--reward_type",
        type=str,
        default="gt",
        choices=[
            "gt", "success_classifier", "weighted", "dense", "inpaint",
            "sparse"
            "blackrobot", "inpaint-blur", "eef_inpaint", "dontcare"
        ],
    )
    # for use with inpaint blur
    parser.add_argument("--most_recent_background",
                        type=str2bool,
                        default=False)
    # inpaint-blur
    parser.add_argument("--blur_sigma", type=float, default=10)
    parser.add_argument("--unblur_cost_scale", type=float, default=3)
    # switch at step L - unblur_timestep
    parser.add_argument("--unblur_timestep", type=float, default=1)

    # control algorithm
    parser.add_argument(
        "--mbrl_algo",
        type=str,
        default="cem",
        choices=["cem"],
    )

    # training
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--record_trajectory", type=str2bool, default=False)
    parser.add_argument("--record_trajectory_interval", type=int, default=5)
    parser.add_argument("--record_video_interval", type=int, default=1)

    # environment
    parser.add_argument("--env", type=str, default="FetchPush")
    args, unparsed = parser.parse_known_args()

    add_prediction_arguments(parser)
    add_dataset_arguments(parser)
    add_cost_arguments(parser)

    if args.mbrl_algo == "cem":
        add_cem_arguments(parser)

    # env specific args
    if args.env == "FetchPush":
        add_fetch_push_arguments(parser)

    return parser


# Env Hyperparameters
def add_fetch_push_arguments(parser: ArgumentParser):
    # override prediction dimension stuff
    parser.set_defaults(robot_dim=6, robot_enc_dim=6)
    parser.add_argument("--img_dim", type=int, default=64)
    parser.add_argument(
        "--camera_name",
        type=str,
        default="external_camera_0",
        choices=[
            "head_camera_rgb", "gripper_camera_rgb", "lidar",
            "external_camera_0"
        ],
    )
    parser.add_argument("--multiview", type=str2bool, default=False)
    parser.add_argument("--camera_ids", type=str2intlist, default=[0, 4])
    parser.add_argument("--pixels_ob", type=str2bool, default=True)
    parser.add_argument("--norobot_pixels_ob", type=str2bool, default=False)
    parser.add_argument("--robot_mask_with_obj", type=str2bool, default=False)
    parser.add_argument("--inpaint_eef", type=str2bool, default=True)
    parser.add_argument("--depth_ob", type=str2bool, default=False)
    parser.add_argument("--object_dist_threshold", type=float, default=0.01)
    parser.add_argument("--gripper_dist_threshold", type=float, default=0.025)
    parser.add_argument("--push_dist", type=float, default=0.2)
    parser.add_argument("--max_episode_length", type=int, default=10)
    parser.add_argument(
        "--robot_goal_distribution",
        type=str,
        default="random",
        choices=["random", "behind_block"],
    )
    parser.add_argument("--large_block", type=str2bool, default=False)
    parser.add_argument("--red_robot", type=str2bool, default=False)
    parser.add_argument("--invisible_demo", type=str2bool, default=False)
    parser.add_argument("--demo_dir", type=str, default="demos/fetch_push")


def add_prediction_arguments(parser):
    parser.add_argument("--lr",
                        default=0.0003,
                        type=float,
                        help="learning rate")
    parser.add_argument("--beta1",
                        default=0.9,
                        type=float,
                        help="momentum term for adam")
    parser.add_argument("--batch_size",
                        default=100,
                        type=int,
                        help="batch size")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="test batch size")
    parser.add_argument("--optimizer",
                        default="adam",
                        help="optimizer to train with")
    parser.add_argument("--niter",
                        type=int,
                        default=300,
                        help="number of epochs to train for")
    parser.add_argument("--epoch_size",
                        type=int,
                        default=600,
                        help="epoch size")
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--dataset",
                        default="smmnist",
                        help="dataset to train with")
    parser.add_argument("--n_past",
                        type=int,
                        default=1,
                        help="number of frames to condition on")
    parser.add_argument(
        "--n_future",
        type=int,
        default=9,
        help="number of frames to predict during training",
    )
    parser.add_argument("--n_eval",
                        type=int,
                        default=10,
                        help="number of frames to predict during eval")
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--rnn_size",
                        type=int,
                        default=256,
                        help="dimensionality of hidden layer")
    parser.add_argument("--prior_rnn_layers",
                        type=int,
                        default=2,
                        help="number of layers")
    parser.add_argument("--posterior_rnn_layers",
                        type=int,
                        default=2,
                        help="number of layers")
    parser.add_argument("--predictor_rnn_layers",
                        type=int,
                        default=2,
                        help="number of layers")
    parser.add_argument("--z_dim",
                        type=int,
                        default=10,
                        help="dimensionality of z_t")
    parser.add_argument(
        "--g_dim",
        type=int,
        default=128,
        help="dimensionality of encoder output vector and decoder input vector",
    )
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--action_enc_dim", type=int, default=2)
    parser.add_argument("--robot_dim", type=int, default=6)
    parser.add_argument("--robot_enc_dim", type=int, default=6)
    parser.add_argument("--robot_joint_dim", type=int, default=7)

    parser.add_argument("--beta",
                        type=float,
                        default=0.0001,
                        help="weighting on KL to prior")

    parser.add_argument(
        "--last_frame_skip",
        type=str2bool,
        default=False,
        help=
        "if true, skip connections go between frame t and frame t+t rather than last ground truth frame",
    )

    parser.add_argument("--model",
                        default="svg",
                        choices=["svg", "det", "copy"])
    parser.add_argument("--model_use_mask", type=str2bool, default=True)
    parser.add_argument("--model_use_robot_state", type=str2bool, default=True)
    parser.add_argument("--reconstruction_loss",
                        default="mse",
                        choices=["mse", "l1", "dontcare_mse"])
    parser.add_argument("--scheduled_sampling", type=str2bool, default=False)
    parser.add_argument("--robot_pixel_weight",
                        type=float,
                        default=0,
                        help="weighting on robot pixels")

    parser.add_argument("--learned_robot_model", type=str2bool, default=False)
    parser.add_argument("--robot_model_ckpt", type=str, default=None)
    parser.add_argument("--use_xy_channel", type=str2bool, default=False)
    parser.add_argument("--load_checkpoint", type=str, default=None)


def add_dataset_arguments(parser):
    parser.add_argument("--data_threads",
                        type=int,
                        default=5,
                        help="number of data loading threads")
    parser.add_argument("--data_root",
                        default="data",
                        help="root directory for data")
    parser.add_argument("--train_val_split", type=float, default=0.8)
    # data collection policy arguments
    parser.add_argument("--temporal_beta", type=float, default=1)
    parser.add_argument("--demo_length", type=int, default=12)
    parser.add_argument("--action_noise", type=float, default=0)
    parser.add_argument(
        "--video_type",
        default="object_inpaint_demo",
        choices=["object_inpaint_demo", "robot_demo", "object_only_demo"],
    )
    # robonet video prediction dataset arguments
    parser.add_argument(
        "--video_length",
        type=int,
        default=31,
        help="max length of the video, used for evaluation dataloader")
    parser.add_argument("--impute_autograsp_action",
                        type=str2bool,
                        default=True)
    parser.add_argument("--preload_ram", type=str2bool, default=False)
    parser.add_argument("--training_regime",
                        type=str,
                        choices=[
                            "multirobot", "singlerobot", "finetune",
                            "train_sawyer_multiview", "finetune_sawyer_view",
                            "finetune_widowx"
                        ],
                        default="multirobot")
    parser.add_argument(
        "--preprocess_action",
        type=str,
        choices=["raw", "camera_raw", "state_infer", "camera_state_infer"],
        default="raw")
    parser.add_argument("--img_augmentation", type=str2bool, default=False)
    parser.add_argument("--color_jitter_range", type=float, default=0.1)
    parser.add_argument("--random_crop_size", type=int, default=59)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--world_error_dict", type=str, default=None)
    parser.add_argument("--finetune_num_train", type=int, default=400)
    parser.add_argument("--finetune_num_test", type=int, default=100)
    parser.add_argument("--num_train", type=int, default=400)
    parser.add_argument("--num_test", type=int, default=100)


# CEM Hyperparameters
def add_cem_arguments(parser):
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--opt_iter", type=int, default=10)
    parser.add_argument("--action_candidates", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--replan_every", type=int, default=1)
    parser.add_argument("--dynamics_model_ckpt", type=str, default=None)
    parser.add_argument("--candidates_batch_size", type=int, default=200)
    parser.add_argument("--use_env_dynamics", type=str2bool, default=False)
    parser.add_argument("--debug_trajectory_path", type=str, default=None)
    parser.add_argument("--debug_cem", type=str2bool, default=False)
    parser.add_argument("--object_demo_dir", type=str, default=None)
    parser.add_argument("--subgoal_start", type=int, default=0)
    parser.add_argument("--sequential_subgoal", type=str2bool, default=True)
    parser.add_argument("--demo_cost", type=str2bool, default=False)
    parser.add_argument("--demo_timescale", type=int, default=1)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument(
        "--demo_type",
        default="object_only_demo",
        choices=["object_inpaint_demo", "object_only_demo", "robot_demo"],
    )
    parser.add_argument("--cem_init_std", type=float, default=1)
    parser.add_argument("--sparse_cost", type=str2bool, default=False)


# Cost Fn Hyperparameters


def add_cost_arguments(parser):
    # cost thresholds for determining goal success
    parser.add_argument("--world_cost_success", type=float, default=4000)
    parser.add_argument("--robot_cost_success", type=float, default=0.01)
    # weight of the costs
    parser.add_argument("--robot_cost_weight", type=float, default=0)
    parser.add_argument("--world_cost_weight", type=float, default=1)
    # checks if pixel diff > threshold before counting it
    parser.add_argument("--img_cost_threshold", type=float, default=None)
    # only used by img don't care cost, divide by number of world pixels
    parser.add_argument("--img_cost_world_norm", type=str2bool, default=True)


def add_ensemble_arguments(parser):
    parser.add_argument("--load_traj_length", type=int, default=10)
    parser.add_argument("--num_ensembles", type=int, default=10)
    parser.add_argument("--train_data_per_ensemble", type=int, default=500)


def argparser():
    """ Directly parses the arguments. """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()
    assert len(unparsed) == 0, unparsed
    return args, unparsed
