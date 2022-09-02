from allensdk.core.reference_space_cache import ReferenceSpaceCache
import re
import nrrd
import os
import requests

class VBN_CCF:

    def __init__(self, manifest_path='manifest.json', resolution=10):
        reference_space_key = 'annotation/ccf_2017'
        self.resolution = resolution
        self.rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=manifest_path)
        # ID 1 is the adult mouse structure graph
        self.tree = self.rspc.get_structure_tree(structure_graph_id=1) 
        self._annotation = None
        self._streamlines = None
        self._manifest_directory = os.path.dirname(manifest_path)


    @property
    def annotation(self):
        if self._annotation is None:
            self._annotation, meta = self.rspc.get_annotation_volume()
        return self._annotation

    
    @property
    def streamlines(self):
        if self._streamlines is None:
            streamlines_path = os.path.join(self._manifest_directory, 'laplacian_10.nrrd')
            #First check to see if the streamlines have already been downloaded
            if os.path.exists(streamlines_path):
                self._streamlines, header = nrrd.read(streamlines_path)
            
            #Otherwise download it and write it to file
            else:
                s = requests.get('https://www.dropbox.com/sh/7me5sdmyt5wcxwu/AACFY9PQ6c79AiTsP8naYZUoa/laplacian_10.nrrd?dl=1')
                with open(streamlines_path, 'wb') as f:
                    f.write(bytes(s.content))
                self._streamlines, header = nrrd.read(streamlines_path)
                
        return self._streamlines


    def get_structure_by_acronym(self, acronym):
        try:
            structure = self.tree.get_structures_by_acronym([acronym])
        except KeyError:
            print(f'Could not find structure corresponding to acronym {acronym}')
            structure = [{}]
        return structure


    def get_structure_name_by_acronym(self, acronym):
        structure = self.get_structure_by_acronym(acronym)[0]
        return structure.get('name', None)

    
    def get_structure_id_by_coordinate(self, ap_coord, dv_coord, lr_coord):

        volume_coords = [int(coord/self.resolution) for coord in [ap_coord, dv_coord, lr_coord]]
        shape = self.annotation.shape
        if any([(v<0 or v>=s) for v,s in zip(volume_coords,shape)]):
            print(f'Coordinate {[ap_coord, dv_coord, lr_coord]} is outside ccf')
            id = 0
        else:
            id = self.annotation[volume_coords[0], volume_coords[1], volume_coords[2]]
        return id


    def get_structure_by_id(self, id):
        if id == 0:
            structure = [{'name': 'outside_brain',
                        'acronym': 'outside_brain'}]
        else:
            structure = self.tree.get_structures_by_id([id])
        return structure


    def get_structure_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        try:
            id = self.get_structure_id_by_coordinate(ap_coord, dv_coord, lr_coord)
            structure = self.get_structure_by_id(id)
        except Exception as e:
            print(f'Could not get structure corresponding to id: {id} due to error {e}')
            structure = [{}]
        return structure


    def get_structure_acronym_by_coordinate(self, ap_coord, dv_coord, lr_coord):

        structure = self.get_structure_by_coordinate(ap_coord, dv_coord, lr_coord)[0]
        return structure.get('acronym', None)

    
    def get_layer_name(self, acronym):

        if acronym in ['CA1', 'CA2', 'CA3']:
            return ''
    
        try:
            first_num = re.findall(r'\d+', acronym)[0]
            first_num_ind = acronym.find(first_num)
            if first_num_ind<0:
                return ''
            
            layer = acronym[first_num_ind:]
            return layer

        except IndexError:
            return ''
    

    def get_cortical_depth_by_coordinate(self, ap_coord, dv_coord, lr_coord):
        volume_coords = [int(coord/10) for coord in [ap_coord, dv_coord, lr_coord]]
        try:
            cortical_depth = self.streamlines[volume_coords[0], volume_coords[1], volume_coords[2]]
        except IndexError as e:
            print(f'Coordinate {[ap_coord, dv_coord, lr_coord]} is outside of CCF')
            cortical_depth = 0
        return cortical_depth