import os
import pickle
import logging
import numpy as np

import torch
import torch.utils.data

from utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    get_joints_from_pose

def save_obj(filename, verts, faces):
    with open(filename, 'w') as file:
        # Write vertices to file
        for vert in verts:
            file.write('v {} {} {}\n'.format(*vert))
        
        # Write faces to file
        for face in faces:
            # Adding 1 to each vertex index, because OBJ files are 1-indexed
            file.write('f')
            for vertex_index in face:
                file.write(' {}'.format(vertex_index + 1))
            file.write('\n')

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            bgcolor=None,
    ):
        self.cfg = {
            'bbox_offset': 0.3,
            'resize_img_scale': [0.5, 0.5],
        }

        logging.info(f'[Dataset Path]: {dataset_path}')

        self.dataset_path = dataset_path

        self.canonical_joints, self.canonical_bbox, self.canonical_vertex, self.canonical_lbs_weights, self.edges, self.faces, self.canonical_pose, self.avg_beta = \
            self.load_canonical_joints()

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        canonical_vertex = cl_joint_data['vertex'].astype('float32')
        canonical_lbs_weights = cl_joint_data['weights'].astype('float32')

        if 'edges' in cl_joint_data:
            canonical_edges = cl_joint_data['edges'].astype(int)
        else:
            canonical_edges = None

        if 'faces' in cl_joint_data:
            canonical_faces = cl_joint_data['faces']
        else:
            canonical_faces = None

        canonical_pose = cl_joint_data['pose'].astype('float32')
        canonical_beta = cl_joint_data['beta'].astype('float32')
            
        # save_obj('./canonical_mesh.obj', canonical_vertex, canonical_faces)

        # # for a-pose
        # import smplx
        # MODEL_DIR = '../utils/smplx/models'
        # sex = 'neutral'
        # smplx_model = smplx.create(MODEL_DIR, model_type='smplx',
        #                  gender=sex, use_face_contour=False,
        #                  num_betas=10,
        #                  ext='npz')
        # # change a-pose here
        # a_pose = torch.from_numpy(np.zeros(smplx_model.NUM_BODY_JOINTS * 3, ).astype(np.float32)).unsqueeze(0)
        # output = smplx_model(betas=torch.from_numpy(canonical_beta).unsqueeze(0), 
        #                     body_pose=a_pose,
        #                     return_verts=True,
        #                     return_full_pose=True)
        # v = output.vertices.detach().cpu().numpy().squeeze()
        # full_pose = output.full_pose.detach().cpu().numpy().squeeze()
        # num_joints = full_pose.reshape(-1, 3).shape[0]
        # a_template_joints = output.joints.detach().cpu().numpy().squeeze()
        # a_template_joints = a_template_joints[:num_joints, :]

        # canonical_joints = a_template_joints
        # canonical_pose = full_pose


        return canonical_joints, canonical_bbox, canonical_vertex, canonical_lbs_weights, canonical_edges, canonical_faces, canonical_pose, canonical_beta


    def skeleton_to_bbox(self, skeleton):
        min_xyz = np.min(skeleton, axis=0) - self.cfg['bbox_offset']
        max_xyz = np.max(skeleton, axis=0) + self.cfg['bbox_offset']

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': np.zeros(63).astype(np.float32),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    def __len__(self):
        return self.get_total_frames()

    def load_data(self):

        results = {}

        bgcolor = (np.random.rand(3) * 255.).astype('float32')
        results['bgcolor'] = bgcolor / 255.

        dst_poses = self.canonical_pose
        dst_tpose_joints = self.canonical_joints

        global_tfms = np.eye(4)

        results.update({
            'global_tfms': global_tfms
        })

        dst_Rs, dst_Ts = body_pose_to_body_RTs(
                dst_poses, dst_tpose_joints
            )
        cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
        results.update({
            'dst_poses': dst_poses,
            'dst_Rs': dst_Rs,
            'dst_Ts': dst_Ts,
            'cnl_gtfms': cnl_gtfms
        })

        # 1. ignore global orientation
        # 2. add a small value to avoid all zeros
        dst_posevec_69 = dst_poses.reshape(-1)[3:] + 1e-2
        results.update({
            'dst_posevec': dst_posevec_69,
        })

        results.update({
            'joints': get_joints_from_pose(dst_poses, dst_tpose_joints),
            'dst_tpose_joints': dst_tpose_joints,
        })

        # for iteration
        for k in results.keys():
            results[k] = torch.from_numpy(results[k]).unsqueeze(0)

        return results

    def get_canonical_info(self):
        info = {
            'canonical_joints': self.canonical_joints,
            'canonical_bbox': {
                'min_xyz': self.canonical_bbox['min_xyz'],
                'max_xyz': self.canonical_bbox['max_xyz'],
                'scale_xyz': self.canonical_bbox['max_xyz'] - self.canonical_bbox['min_xyz'],
            },
            'canonical_vertex': self.canonical_vertex,
            'canonical_lbs_weights': self.canonical_lbs_weights,
            'edges': self.edges,
            'faces': self.faces,
        }
        return info