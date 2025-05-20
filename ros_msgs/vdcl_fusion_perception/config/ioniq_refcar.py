#ioniq reference car BEVFusion configs

config_path='/home/sgsp/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0100_depth_with_lidar_depth.py'
checkpoint_path='/home/sgsp/mmdetection3d/work_dirs/bevfusion_lidar-cam_voxel0100_depth_with_lidar_depth_400q/epoch_6.pth'

num_sweeps=10

#position:x,y,z
#rotation:roll,pitch,yaw

lidar_parameter={
    "LIDAR_TOP": {
        "lidar_name": "LIDAR_TOP",
        "lidar_type": "LIDAR",
        "lidar_position": [0.0, 0.0, 0.0],
        "lidar_rotation": [0.0, 0.0, 0.0],
        "lidar_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
}

camera_parameter = {
    "CAM_FRONT": {
        "camera_name": "CAM_FRONT",
        "camera_type": "CAMERA",
        "camera_position": [0.0, 0.0, 0.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        },
    "CAM_FRONT_LEFT": {
        "camera_name": "CAM_FRONT_LEFT",
        "camera_type": "CAMERA",
        "camera_position": [0.0, 0.0, 0.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    },
    "CAM_FRONT_RIGHT": {
        "camera_name": "CAM_FRONT_RIGHT",
        "camera_type": "CAMERA",
        "camera_position": [0.0, 0.0, 0.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    },
    "CAM_BACK": {
        "camera_name": "CAM_BACK",
        "camera_type": "CAMERA",
        "camera_position": [0.0, 0.0, 0.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    },
    "CAM_BACK_LEFT": {
        "camera_name": "CAM_BACK_LEFT",
        "camera_type": "CAMERA",
        "camera_position": [0.0, 0.0, 0.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    },
    "CAM_BACK_RIGHT": {
        "camera_name": "CAM_BACK_RIGHT",
        "camera_type": "CAMERA",
        "camera_position": [0.0, 0.0, 0.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "camera_intrinsic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
}
