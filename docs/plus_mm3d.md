pip install Shapely==1.8.5
pip install similaritymeasures==0.7.0

mmdetection3d/mmdet3d/datasets/pipelines/formating.py
注释掉
# for key in ['lidar2img', 'lidar2camera', 'camera_intrinsics']:
#     if key not in results:
#         continue
#     results[key] = DC(to_tensor(results[key]), stack=True)

mmdetection3d/mmdet3d/core/bbox/assigners/hungarian_assigner.py
注释掉class BBox3DL1Cost(object)和HungarianAssigner3D的注册
# @MATCH_COST.register_module()
# @BBOX_ASSIGNERS.register_module()
