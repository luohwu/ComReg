import copy

import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh
import open3d as o3d

from utility.AverageMeter import AverageMeter
from utility.converter import *
from utility.us_denoise import denoise_dbscan_cloud
import time

from torch.utils.data import Dataset
from utility import misc
from utility.converter import to_o3d_pcd
# PCA-axis generator removed; synthesis now lives in
# generate_partial_pcd/us_pcd_synthesis_real_trajectory.py (this import was unused here).
######Load data############
class DatasetCoupled(Dataset):
    def __init__(self, conf):
        super(DatasetCoupled, self).__init__()
        self.device = torch.device('cpu')
        pass
        self.conf = conf
        self.specimen_folder = os.path.join(conf['dataset_root_folder'], f"specimen{conf['specimen_id']:02d}")
        self.ultrasound_records_folder = os.path.join(self.specimen_folder, "ultrasound_records", conf['anatomy'])
        self.raw_data_file = os.path.join(conf['dataset_root_folder'], f"specimen{conf['specimen_id']:02d}",
                                          "CT_bone_segmentations", f"{conf['anatomy']}.stl")
        self.processed_data_file = self.raw_data_file.replace(".stl", ".pt")
        if os.path.exists(self.processed_data_file):
            print(self.processed_data_file)
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            self.process_data()
        self.npoints_gt = conf['npoints_gt']
        self.npoints_input = conf['npoints_input']
        self.dataset_len = conf.get('dataset_len', 2048)
        self.max_rotation_deg = conf.get('max_rotation_deg', 90.0)
        self.max_translation = conf.get('max_translation', 0.5)

        print("Loading from saved data...")
        prep_data = torch.load(self.processed_data_file, map_location=self.device,
                               weights_only=False)
        #
        self.sample_near = prep_data["sample_near"]
        self.sample_near_normal = prep_data["sample_near_normal"]
        self.sample = prep_data["query_points"]
        self.pcd_gt = prep_data["pointcloud"]
        self.pcd_gt = self.pcd_gt[torch.randperm(self.pcd_gt.shape[0], device=self.device)[:self.npoints_gt]]
        #
        self.pcd_gt_normals = prep_data["pointcloud_normals"]
        #
        self.shape_scale = prep_data["shape_scale"]
        self.shape_center = prep_data["shape_center"]
        self.grid_sparse_udf_gt = prep_data["grid_sparse_udf_gt"]
        self.grid_sparse = prep_data["grid_sparse"]
        #
        self.sample_points_num = self.sample.shape[0] - 1
        self.object_bbox_min, _ = torch.min(self.sample_near, dim=0)
        self.object_bbox_min = self.object_bbox_min - 0.05
        self.object_bbox_max, _ = torch.max(self.sample_near, dim=0)
        self.object_bbox_max = self.object_bbox_max + 0.05
        print('Data bounding box:', self.object_bbox_min, self.object_bbox_max)
        print('NP Load data: End')
        self.specimen_id = conf['specimen_id']
        self.anatomy = conf['anatomy']

        intra_pcds = []
        for record_folder_name in os.listdir(self.ultrasound_records_folder):
            record_folder = os.path.join(self.ultrasound_records_folder, record_folder_name)
            if os.path.isfile(record_folder):
                continue
            intra_pcd_o3d = o3d.io.read_point_cloud(
                os.path.join(record_folder, "3D_reconstructions", "with_pred_labels",
                             "reconstruction_pcd_filtered.xyz")).farthest_point_down_sample(self.npoints_input)
            # prep_data_o3d = o3d.geometry.PointCloud(
            #     o3d.utility.Vector3dVector((self.pcd_gt * self.shape_scale + self.shape_center).cpu().numpy()))
            # CD = 0.0 * np.asarray(prep_data_o3d.compute_point_cloud_distance(intra_pcd_o3d)).mean() + 0.5 * np.asarray(
            #     intra_pcd_o3d.compute_point_cloud_distance(prep_data_o3d)).mean()
            # print(f"CD: {CD:.3f}")
            # o3d.visualization.draw_geometries([prep_data_o3d,intra_pcd_o3d])
            intra_pcd = np.asarray(intra_pcd_o3d.points)
            intra_pcds.append(intra_pcd)

        self.intra_pcds_mm = torch.tensor(intra_pcds).float()

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num // 10, batch_size, replace=False)
        index = index_fine * 10 + index_coarse
        samples = self.sample[index]
        samples_near = self.sample_near[index]
        samples_near_normal = self.sample_near_normal[index]

        return samples, samples_near, samples_near_normal, self.pcd_gt, self.pcd_gt_normals

    def __len__(self):
        return self.dataset_len

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        # centroid = torch.mean(pc, dim=0)
        # pc = pc - centroid
        # m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        # pc = pc / m
        return pc

    def __getitem__(self, idx):
        disturbance_r = np.random.uniform(-90, 90, 3)
        disturbance_t = np.random.uniform(-0.5, 0.5, 3)
        # print(f"disturbance t: {disturbance_t}, rotation: {disturbance_r}")
        Transformation_matrix_disturbance = torch.tensor(vectorToMatrix(t=disturbance_t, rotation_vector=disturbance_r)).float().to(self.device)
        gt_downsampled=self.pcd_gt[torch.randperm(self.pcd_gt.shape[0], device=self.device)[:self.npoints_gt]]
        gt_downsampled_transformed=torch.matmul(Transformation_matrix_disturbance[:3,:3],gt_downsampled.transpose(0,1))+Transformation_matrix_disturbance[:3,3].unsqueeze(-1)
        #
        gt_downsampled_transformed = self.pc_norm(gt_downsampled_transformed.transpose(0,1))
        # partial, _ = misc.seprate_point_cloud(gt_downsampled_transformed, self.npoints_gt, [int(self.npoints_gt * 1 / 4), int(self.npoints_gt * 3 / 4)],
        #                                       fixed_points=None)
        #
        return gt_downsampled_transformed


    ########Convert the .ply file into .npz ############
    def process_data(self):
        assert os.path.isfile(self.raw_data_file)
        
        if ".stl" in self.raw_data_file:
            mesh=o3d.io.read_triangle_mesh(self.raw_data_file)
            pointcloud=np.asarray(mesh.sample_points_uniformly(100000).points)
        elif ".xyz" in self.raw_data_file:
            pointcloud=o3d.io.read_point_cloud(self.raw_data_file)
            pointcloud=np.asarray(pointcloud.points)
        else:
            print('Only support .xyz or .stl data. Please make adjust your data.')
            exit()

        shape_scale = np.max(
            [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
             np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
        shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                        (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                        (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
        self.shape_scale = shape_scale
        self.shape_center = shape_center
        pointcloud = pointcloud - shape_center
        pointcloud = pointcloud / shape_scale

        pcd_np, normals_np = self.FPS_sampling(pointcloud)

        pointcloud = torch.from_numpy(pcd_np).to(self.device).float()
        pointcloud_normals = torch.from_numpy(normals_np).to(self.device).float()

        grid_samp = 30000

        def gen_grid(start, end, num):
            x = np.linspace(start, end, num=num)
            y = np.linspace(start, end, num=num)
            z = np.linspace(start, end, num=num)
            g = np.meshgrid(x, y, z)
            positions = np.vstack([np.ravel(arr) for arr in g])
            return positions.swapaxes(0, 1)

        dot5 = gen_grid(-0.5, 0.5, 70)
        dot10 = gen_grid(-1.0, 1.0, 50)
        grid_sparse = gen_grid(-1.0, 1.0, 10)
        grid_sparse_udf_gt = self.chamfer_distance(grid_sparse, pcd_np)

        grid_sparse = torch.from_numpy(grid_sparse).to(self.device).float()
        grid_sparse_udf_gt = torch.from_numpy(grid_sparse_udf_gt).to(self.device).float()

        grid = np.concatenate((dot5, dot10))
        # grid = dot5
        grid = torch.from_numpy(grid).to(self.device).float()
        grid_f = grid[torch.randperm(grid.shape[0])[0:grid_samp]]

        query_per_point = 20
        query_points = self.sample_query2(query_per_point, pointcloud, pcd_np)

        # concat sampled points with grid points
        query_points = torch.cat([query_points, grid_f]).float()

        ## find nearest neiboring point cloud for each query point
        POINT_NUM = 1000  # divide by batch to avoid out-of-memory
        if query_points.shape[0] % POINT_NUM > 0:
            query_points = query_points[:-(query_points.shape[0] % POINT_NUM), :]
        query_points_nn = torch.reshape(query_points, (-1, POINT_NUM, 3))
        sample_near_tmp = []
        sample_near_normal_temp = []
        for j in range(query_points_nn.shape[0]):
            nearest_idx = self.search_nearest_point(torch.tensor(query_points_nn[j]).float().cuda(),
                                                    torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = nearest_points.reshape(-1, 3)
            sample_near_tmp.append(nearest_points)

            nearest_normals = pointcloud_normals[nearest_idx]
            nearest_normals = nearest_normals.reshape(-1, 3)
            sample_near_normal_temp.append(nearest_normals)

        sample_near_tmp = torch.stack(sample_near_tmp, 0)
        sample_near_tmp = sample_near_tmp.reshape(-1, 3)
        sample_near = sample_near_tmp

        sample_near_normal_temp = torch.stack(sample_near_normal_temp, 0)
        sample_near_normal_temp = sample_near_normal_temp.reshape(-1, 3)
        sample_near_normal = sample_near_normal_temp

        print("Saving files...")
        torch.save({
            "shape_scale": shape_scale,
            "shape_center": shape_center,
            "pointcloud": pointcloud,
            "pointcloud_normals": pointcloud_normals,
            "query_points": query_points,
            "sample_near": sample_near,
            "sample_near_normal": sample_near_normal,
            "grid_sparse_udf_gt": grid_sparse_udf_gt,
            "grid_sparse": grid_sparse,
        },
            self.processed_data_file
        )

    def chamfer_distance(self, A, B):
        # Calculate the squared Euclidean distance between each pair of points
        # Result is an N x M matrix where entry (i, j) is the squared distance between A[i] and B[j]
        distances = np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2)

        # Find the minimum distance for each point in A to any point in B
        min_distances_A_to_B = np.min(distances, axis=1)

        # Return the array of Chamfer distances for each point in A to B
        return min_distances_A_to_B

    def FPS_sampling(self, point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=1000))

        pcd_concat = o3d.geometry.PointCloud()

        if len(pcd.points) > 60000:
            # Perform farthest point sampling on the original point cloud
            pcd_down_1 = pcd.farthest_point_down_sample(5000*2)  # First sampling
            pcd_down_2 = pcd.farthest_point_down_sample(15000*2)  # Second sampling

            # Concatenate sampled points and their normals
            points_concat = np.concatenate(
                (np.asarray(pcd_down_1.points), np.asarray(pcd_down_2.points)), axis=0
            )
            normals_concat = np.concatenate(
                (np.asarray(pcd_down_1.normals), np.asarray(pcd_down_2.normals)), axis=0
            )
        else:
            # Use all points and normals directly if the cloud is small
            points_concat = np.asarray(pcd.points)
            normals_concat = np.asarray(pcd.normals)

            # Set points and normals to the concatenated point cloud
        pcd_concat.points = o3d.utility.Vector3dVector(points_concat.reshape(-1, 3))
        pcd_concat.normals = o3d.utility.Vector3dVector(normals_concat.reshape(-1, 3))
        pcd_concat.orient_normals_consistent_tangent_plane(15)

        # (Optional) Visualize the concatenated point cloud
        # o3d.visualization.draw_geometries([pcd_concat])

        print("Number of points:", len(pcd_concat.points))
        point_cloud_concat = np.asarray(pcd_concat.points)
        normals_concat = np.asarray(pcd_concat.normals)

        return point_cloud_concat, normals_concat

    # find the 50th nearest neighbor for each point in pc
    # this will be the std for the gaussian for generating query
    def sample_query(self, query_per_point, pc):

        # scale = 0.25

        dists = torch.cdist(pc, pc)

        std, _ = torch.topk(dists, 50, dim=-1, largest=False)  # shape: 1024, 50

        std = std[:, -1].unsqueeze(-1)  # new shape is 1024, 1

        query_points = torch.empty(size=(pc.shape[0] * query_per_point, 3)).to(self.device).float()
        count = 0

        for idx, p in enumerate(pc):
            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                 size=(query_per_point, 3)).to(self.device).float()

            # query locations in space
            q = p + q_loc

            query_points[count:count + query_per_point] = q

            count += query_per_point

        return query_points

    def sample_query2(self, query_per_point, pc, pointcloud):
        # divide point cloud by bacth to avoid out-of-memory

        ptree = cKDTree(pointcloud)
        std = []
        for idx_progress, p in enumerate(np.array_split(pointcloud, 100, axis=0)):
            # print(idx_progress)
            d = ptree.query(p, 51)
            std.append(d[0][:, -1])

        std = np.concatenate(std)

        std = torch.from_numpy(std).to(self.device).float()

        std = std.unsqueeze(-1)  # new shape is 1024, 1

        query_points = torch.empty(size=(pc.shape[0] * query_per_point, 3)).to(self.device).float()
        count = 0

        for idx, p in enumerate(pc):
            # print(f"working on {idx}/{len(pc)}")

            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                 size=(query_per_point, 3)).to(self.device).float()

            # query locations in space
            q = p + q_loc

            query_points[count:count + query_per_point] = q

            count += query_per_point

        return query_points

    # the closest point in the pc for all query points
    def find_nearest_query_neighbor(self, pc, query_points):

        dists = torch.cdist(query_points, pc).detach().cpu().numpy()
        min_dist, min_idx = torch.min(dists, dim=-1).detach().cpu().numpy()
        nearest_neighbors = pc[min_idx]

        return nearest_neighbors, min_dist.unsqueeze(-1)

    ######Find the nearest neighbour points ##########
    def search_nearest_point(self, point_batch, point_gt):
        num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
        point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
        point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

        distances = torch.sqrt(torch.sum((point_batch - point_gt) ** 2, axis=-1) + 1e-12)
        dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

        return dis_idx


class DatasetCoupled_slice_wise(DatasetCoupled):
    def __init__(self, conf):
        super(DatasetCoupled, self).__init__()
        self.device = torch.device('cpu')
        self.conf = conf
        self.specimen_folder = os.path.join(conf['dataset_root_folder'], f"specimen{conf['specimen_id']:02d}")
        self.ultrasound_records_folder = os.path.join(self.specimen_folder, "ultrasound_records", conf['anatomy'])
        self.raw_data_file = os.path.join(conf['dataset_root_folder'], f"specimen{conf['specimen_id']:02d}",
                                          "CT_bone_segmentations", f"{conf['anatomy']}.stl")
        self.processed_data_file = self.raw_data_file.replace(".stl", ".pt")
        if os.path.exists(self.processed_data_file):
            print(self.processed_data_file)
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            self.process_data()
        self.npoints_gt = conf['npoints_gt']
        self.npoints_input = conf['npoints_input']
        self.dataset_len = conf.get('dataset_len', 2048)
        self.max_rotation_deg = conf.get('max_rotation_deg', 90.0)
        self.max_translation = conf.get('max_translation', 0.5)

        print("Loading from saved data...")
        prep_data = torch.load(self.processed_data_file, map_location=self.device,
                               weights_only=False)
        #
        self.sample_near = prep_data["sample_near"]
        self.sample_near_normal = prep_data["sample_near_normal"]
        self.sample = prep_data["query_points"]
        self.pcd_gt = prep_data["pointcloud"]
        self.pcd_gt = self.pcd_gt[torch.randperm(self.pcd_gt.shape[0], device=self.device)[:self.npoints_gt]]
        #
        self.pcd_gt_normals = prep_data["pointcloud_normals"]
        #
        self.shape_scale = torch.tensor(prep_data["shape_scale"])
        self.shape_center = torch.tensor(prep_data["shape_center"])
        self.grid_sparse_udf_gt = prep_data["grid_sparse_udf_gt"]
        self.grid_sparse = prep_data["grid_sparse"]
        #
        self.sample_points_num = self.sample.shape[0] - 1
        self.object_bbox_min, _ = torch.min(self.sample_near, dim=0)
        self.object_bbox_min = self.object_bbox_min - 0.05
        self.object_bbox_max, _ = torch.max(self.sample_near, dim=0)
        self.object_bbox_max = self.object_bbox_max + 0.05
        print('Data bounding box:', self.object_bbox_min, self.object_bbox_max)
        print('NP Load data: End')
        self.specimen_id = conf['specimen_id']
        self.anatomy = conf['anatomy']

        intra_pcds = []
        # Optional DBSCAN denoising of the real US clouds (removes clustered
        # segmentation false-positives; tuned constant config in utility.us_denoise).
        # Off by default so the original pipeline is unchanged; enabled via the
        # `denoise_intra` dataset conf key for A/B comparison. Runs on the raw cloud
        # (before FPS) so the FPS target stays well above npoints_input.
        denoise_intra = bool(self.conf.get('denoise_intra', False))
        for record_folder_name in os.listdir(self.ultrasound_records_folder):
            record_folder = os.path.join(self.ultrasound_records_folder, record_folder_name)
            if not os.path.isdir(record_folder):
                continue
            intra_pcd_o3d = o3d.io.read_point_cloud(
                os.path.join(record_folder, "3D_reconstructions", "with_pred_labels",
                             "reconstruction_pcd_filtered.xyz"))
            if denoise_intra:
                intra_pcd_o3d = denoise_dbscan_cloud(intra_pcd_o3d)
            intra_pcd_o3d = intra_pcd_o3d.farthest_point_down_sample(self.npoints_input)
            # prep_data_o3d=o3d.geometry.PointCloud(o3d.utility.Vector3dVector((self.pcd_gt*self.shape_scale+self.shape_center).cpu().numpy()))
            # CD=0.0*np.asarray(prep_data_o3d.compute_point_cloud_distance(intra_pcd_o3d)).mean()+0.5*np.asarray(intra_pcd_o3d.compute_point_cloud_distance(prep_data_o3d)).mean()
            # print(f"CD: {CD:.3f}")
            # o3d.visualization.draw_geometries([prep_data_o3d,intra_pcd_o3d])
            intra_pcd = np.asarray(intra_pcd_o3d.points)
            intra_pcds.append(intra_pcd)

        self.intra_pcds_mm = torch.tensor(intra_pcds).float()
        self.intra_pcd_merged_mm=torch.tensor(np.concatenate(intra_pcds)).float()
        intra_pcd_merged_mm_o3d=to_o3d_pcd(self.intra_pcd_merged_mm).farthest_point_down_sample(self.npoints_input)
        self.intra_pcd_merged_mm=torch.tensor(np.asarray(intra_pcd_merged_mm_o3d.points)).float()

        self.CT_bone_mesh_o3d_mm = None
        self.intra_pcds_o3d_mm = None
        self.best_azimuth = None
        # Which simulated-data file to load. Defaults to the original
        # hand-trajectory output; set `simulated_data_suffix` in the dataset conf
        # to switch generators (e.g. the real-trajectory stochastic simulator)
        # without disturbing the existing pipeline.
        sim_suffix = self.conf.get('simulated_data_suffix', "_simulated_slice_data.npy")
        simulated_data_file = self.raw_data_file.replace(".stl", sim_suffix)
        # Auto-generate the simulated US data on first use, so training/testing
        # never requires running us_pcd_synthesis_real_trajectory.py by hand.
        if not os.path.isfile(simulated_data_file):
            print("generating simulated US data "
                  f"(missing {os.path.basename(simulated_data_file)})...")
            from generate_partial_pcd.us_pcd_synthesis_real_trajectory import ensure_simulated_data
            ensure_simulated_data(
                self.conf['specimen_id'], self.conf['anatomy'],
                dataset_root_folder=self.conf['dataset_root_folder'],
                output_suffix=sim_suffix)
        temp = np.load(simulated_data_file, allow_pickle=True).item()
        self.intra_pcds_simulated_mm_list = temp['pcd_simulated_list']
        # self.check_dataset()

    def check_dataset(self):
        AverageMeter_diff=AverageMeter()
        for i in range(10):
            pcd_merged = ((self.intra_pcd_merged_mm - self.shape_center) / self.shape_scale).float()
            disturbance_r = np.random.uniform(-90 * 1, 90 * 1, 3)
            disturbance_t = np.random.uniform(-0.5, 0.5, 3)
            # print(f"disturbance t: {disturbance_t}, rotation: {disturbance_r}")
            Transformation_matrix_disturbance = vectorToMatrix(t=disturbance_t, rotation_vector=disturbance_r)
            Transformation_matrix_disturbance_inv = invert_transformation_matrix(Transformation_matrix_disturbance)
            Transformation_matrix_disturbance = torch.tensor(Transformation_matrix_disturbance).float().to(self.device)
            Transformation_matrix_disturbance_inv = torch.tensor(Transformation_matrix_disturbance_inv).float().to(
                self.device)
            pcd_merged_transformed = torch.matmul(Transformation_matrix_disturbance[:3, :3],
                                                  pcd_merged.transpose(0, 1)) + \
                                     Transformation_matrix_disturbance[:3, 3].unsqueeze(-1)
            AverageMeter_diff.update(torch.linalg.norm(
                Transformation_matrix_disturbance[:3, 3] - pcd_merged_transformed.T.mean(dim=0)) * self.shape_scale)
        print(f"centroid error:{AverageMeter_diff.avg()}")


    def __getitem__(self, idx):
        # intra_pcds_simulated_mm_list pools the simulated clouds from ALL records
        # of this anatomy. Each record is one real probe sweep = one global sweep
        # direction, so drawing uniformly from this list samples across the
        # MULTIPLE global sweep directions available for the anatomy (per-sweep
        # tilt / speed / dropped-slice perturbations are already baked into each
        # precomputed cloud).
        pcd_simulated_np_mm = self.intra_pcds_simulated_mm_list[
            np.random.randint(len(self.intra_pcds_simulated_mm_list))]
        idx = np.random.permutation(len(pcd_simulated_np_mm))[:self.conf['npoints_input']]
        pcd_simulated_np_mm = pcd_simulated_np_mm[idx, :]
        pcd_simulated_np = (pcd_simulated_np_mm - self.shape_center.cpu().numpy()) / self.shape_scale.cpu().numpy()
        pcd_simulated_tensor = torch.tensor(pcd_simulated_np).float().to(self.device)




        # Random axis-angle disturbance: rotation of up to +-max_rotation_deg about a
        # uniformly random axis, with translation along that same axis in [-max_translation, max_translation].
        Transformation_matrix_disturbance = random_axis_angle_transform(
            angle_deg_bound=self.max_rotation_deg, t_bound=self.max_translation)
        Transformation_matrix_disturbance_inv = invert_transformation_matrix(Transformation_matrix_disturbance)
        Transformation_matrix_disturbance = torch.tensor(Transformation_matrix_disturbance).float().to(self.device)
        Transformation_matrix_disturbance_inv = torch.tensor(Transformation_matrix_disturbance_inv).float().to(
            self.device)

        gt_transformed = torch.matmul(Transformation_matrix_disturbance[:3, :3], self.pcd_gt.transpose(0, 1)) + \
                         Transformation_matrix_disturbance[:3, 3].unsqueeze(-1)
        gt_transformed = gt_transformed.transpose(0, 1)

        pcd_simulated_transformed = torch.matmul(Transformation_matrix_disturbance[:3, :3],
                                                 pcd_simulated_tensor.transpose(0, 1)) + \
                                    Transformation_matrix_disturbance[:3, 3].unsqueeze(-1)


        #
        centroid_pcd_simulated_transformed = pcd_simulated_transformed.mean(dim=1)
        T_normalized = np.eye(4)
        T_normalized[:3, 3] = -centroid_pcd_simulated_transformed
        T_normalized = torch.tensor(T_normalized).float().to(self.device)
        pcd_simulated_transformed = pcd_simulated_transformed - centroid_pcd_simulated_transformed.unsqueeze(-1)
        pcd_simulated_transformed = pcd_simulated_transformed.transpose(0, 1)
        gt_transformed = gt_transformed - centroid_pcd_simulated_transformed.unsqueeze(0)
        Transformation_matrix_disturbance_inv = torch.matmul(Transformation_matrix_disturbance_inv,
                                                             torch.linalg.inv(T_normalized))
        # print(pcd_simulated_transformed.shape)
        # o3d.visualization.draw_geometries([to_o3d_pcd(gt_transformed).paint_uniform_color([1,0,0]),to_o3d_pcd(pcd_simulated_transformed).paint_uniform_color([0,1,0])])
        # print(f"gt_transformed:{gt_transformed.shape}, pcd_simulated_transformed: {pcd_simulated_transformed.shape}")
        return gt_transformed, pcd_simulated_transformed, self.pcd_gt, Transformation_matrix_disturbance_inv


if __name__=="__main__":
    from utility.read_confs import read_confs
    conf_file= './test.conf'
    conf=read_confs(conf_file)
    conf.put('dataset.specimen_id', 1)
    conf.put('dataset.anatomy', 'fibula')
    conf.put('dataset.intra_scan_ids', [1,2,3,4])

    DatasetCoupled(conf['dataset'])
