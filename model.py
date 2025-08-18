import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


def smiles2fp(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    Chem.SanitizeMol(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, features)
    fingerprint = torch.from_numpy(features).float().view(1,-1)
    return fingerprint ### [1,2048] torch.Tensor

class Ligand2D(nn.Module):
    """
    input: SMILES
    output: scalar
    """
    def __init__(self, ):
        super(Ligand2D, self).__init__()
        self.input_mlp = nn.Linear(2048, 100)
        self.output_mlp = nn.Linear(100, 1)

    def forward(self, smiles_):
        """
            :param smiles_
                - list of SMILES string
                - SMILES string
        """

        if type(smiles_) == list:
            fps = [smiles2fp(s) for s in smiles_]
            fps = torch.cat(fps, 0)
            hidden_state = F.relu(self.input_mlp(fps))
            output = self.output_mlp(hidden_state)
            output = output.view(-1)
            log_output = F.log_softmax(output, dim=0) # Added dim=0 for clarity
            prob_output = F.softmax(output, dim=0)   # Added dim=0 for clarity
            return log_output, prob_output.tolist()
        else:
            ### smiles string
            fingerprint = smiles2fp(smiles_)
            hidden_state = F.relu(self.input_mlp(fingerprint))
            output = self.output_mlp(hidden_state)
            return output ### [1,1]


class Ligand2D_product(nn.Module):
    '''
        input:  ligand2d & product_smiles
        output: scalar
    '''
    def __init__(self, ):
        super(Ligand2D_product, self).__init__()
        self.ligand_mlp = nn.Linear(2048, 100)
        self.product_mlp = nn.Linear(2048, 100)
        self.output_mlp = nn.Linear(200, 1)

    def forward(self, ligand_smiles, product_smiles_list):
        n = len(product_smiles_list)
        ligand_fp = smiles2fp(ligand_smiles)
        ligand_embedding = F.relu(self.ligand_mlp(ligand_fp))
        ligand_embedding = ligand_embedding.repeat(n,1)

        product_fps = [smiles2fp(smiles) for smiles in product_smiles_list]
        product_fps = torch.cat(product_fps, 0)
        product_embeddings = F.relu(self.product_mlp(product_fps))

        latent_variable = torch.cat([ligand_embedding, product_embeddings], 1)
        output = self.output_mlp(latent_variable).view(-1)
        log_output = F.log_softmax(output, dim=0) # Added dim=0 for clarity
        prob_output = F.softmax(output, dim=0)   # Added dim=0 for clarity
        return log_output, prob_output.tolist()

def atom2int(atom):
    atom_list = ['C', 'N', 'S', 'O', 'H', 'unknown']
    if atom in atom_list:
        return atom_list.index(atom)
    return len(atom_list)-1

def pdbtofeature(pdbfile, centers, pocket_size):
    """ centers=(center_x, center_y, center_z); pocket_size=(size_x, size_y, size_z) """
    with open(pdbfile, 'r') as fin:
        lines = fin.readlines()
    
    features = []
    center_x, center_y, center_z = centers
    size_x, size_y, size_z = pocket_size
    
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                # Using robust fixed-column parsing
                xx = float(line[30:38])
                yy = float(line[38:46])
                zz = float(line[46:54])

                if (center_x - size_x / 2 < xx < center_x + size_x / 2 and
                    center_y - size_y / 2 < yy < center_y + size_y / 2 and
                    center_z - size_z / 2 < zz < center_z + size_z / 2):
                    
                    atom_type = line[76:78].strip()
                    if atom_type == '': # Fallback for atom type
                         atom_type = line[12:16].strip()
                    
                    atom_type_int = atom2int(atom_type)
                    coordinates = torch.FloatTensor([xx, yy, zz]).view(1, -1)
                    features.append((atom_type_int, coordinates))
            except (ValueError, IndexError):
                continue
    
    if not features:
        return torch.LongTensor([]).view(1,-1), torch.FloatTensor([]), torch.ByteTensor([]).view(1,-1)

    atom_idx = torch.LongTensor([feature[0] for feature in features]).view(1,-1)
    mask = torch.ByteTensor([True for feature in features]).view(1,-1)
    positions = torch.cat([feature[1] for feature in features], dim=0)
    return atom_idx, positions, mask

receptor_info_list = [
    ('4r6e', './pdb/4r6e.pdb', -70.76, 21.82, 28.33, 15.0, 15.0, 15.0),
    ('3pbl', './pdb/3pbl.pdb', 9, 22.5, 26, 15, 15, 15),
    ('1iep', './pdb/1iep.pdb', 15.6138918, 53.38013513, 15.454837, 15, 15, 15),
    ('2rgp', './pdb/2rgp.pdb', 16.29212, 34.870818, 92.0353, 15, 15, 15),
    ('3eml', './pdb/3eml.pdb', -9.06363, -7.1446, 55.86259999, 15, 15, 15),
    ('3ny8', './pdb/3ny8.pdb', 2.2488, 4.68495, 51.39820000000001, 15, 15, 15),
    ('4rlu', './pdb/4rlu.pdb', -0.73599, 22.75547, -31.23689, 15, 15, 15),
    ('4unn', './pdb/4unn.pdb', 5.684346153, 18.1917, -7.3715, 15, 15, 15),
    ('5mo4', './pdb/5mo4.pdb', -44.901, 20.490354, 8.48335, 15, 15, 15),
    ('7l11', './pdb/7l11.pdb', -21.81481, -4.21606, -27.98378, 15, 15, 15),
]

receptor2pdbfeature = dict()

for receptor_info in receptor_info_list:
    name_of_receptor, filename_of_receptor, center_x, center_y, center_z, size_x, size_y, size_z = receptor_info
    atom_idx, positions, mask = pdbtofeature(pdbfile=filename_of_receptor,
                                             centers=(center_x, center_y, center_z),
                                             pocket_size=(size_x, size_y, size_z))
    receptor2pdbfeature[name_of_receptor] = (atom_idx, positions, mask)

def pdbqtvina2feature(pdbqt_file):
    """
    【最终修正版】: 健壮地解析.vina文件，只提取第一个模型（最佳姿态）。
    - 不再依赖"MODEL 2"是否存在，而是通过寻找"ENDMDL"来界定第一个模型的范围。
    - 增加了对空文件或解析失败情况的处理。
    """
    try:
        with open(pdbqt_file, 'r') as fin:
            lines = fin.readlines()
    except FileNotFoundError:
        # 如果文件不存在，返回空的张量
        return torch.LongTensor([]).view(1,-1), torch.FloatTensor([]), torch.ByteTensor([]).view(1,-1)

    model_1_lines = []
    in_model_1 = False
    for line in lines:
        # 找到第一个模型的开始
        if line.strip().startswith("MODEL 1"):
            in_model_1 = True
            continue
        # 找到第一个模型的结束
        if line.strip().startswith("ENDMDL"):
            if in_model_1:
                break # 结束循环
        
        if in_model_1 and (line.startswith("ATOM") or line.startswith("HETATM")):
            model_1_lines.append(line)
            
    if not model_1_lines:
        # 如果第一个模型中没有原子行，返回空张量
        return torch.LongTensor([]).view(1,-1), torch.FloatTensor([]), torch.ByteTensor([]).view(1,-1)

    features = []
    for line in model_1_lines:
        try:
            # 使用固定列宽解析，更可靠
            atom_name_raw = line[12:16].strip()
            atom_name = ''.join([i for i in atom_name_raw if not i.isdigit()])
            atom_type = atom2int(atom_name)
            
            xx = float(line[30:38])
            yy = float(line[38:46])
            zz = float(line[46:54])
            coordinates = torch.FloatTensor([xx, yy, zz]).view(1, -1)
            features.append((atom_type, coordinates))
        except (ValueError, IndexError):
            continue

    if not features:
        return torch.LongTensor([]).view(1,-1), torch.FloatTensor([]), torch.ByteTensor([]).view(1,-1)

    atom_idx = torch.LongTensor([feature[0] for feature in features]).view(1,-1)
    mask = torch.ByteTensor([True for feature in features]).view(1,-1)
    positions = torch.cat([feature[1] for feature in features], dim=0)
    return atom_idx, positions, mask

def featurize_receptor_and_ligand(name_of_receptor, pdbqt_file):
    receptor_atom_idx, receptor_positions, receptor_mask = receptor2pdbfeature[name_of_receptor]
    ligand_atom_idx, ligand_positions, ligand_mask = pdbqtvina2feature(pdbqt_file)
    
    # 增加检查，防止空配体导致错误
    if ligand_atom_idx.numel() == 0:
        return None

    atom_idx = torch.cat([receptor_atom_idx, ligand_atom_idx], dim=1)
    positions = torch.cat([receptor_positions, ligand_positions], dim=0)
    positions = torch.unsqueeze(positions, 0)
    mask = torch.cat([receptor_mask, ligand_mask], dim=1)
    return atom_idx, positions, mask

def featurize_receptor_and_ligand_list(name_of_receptor, pdbqt_file_list):
    receptor_atom_idx, receptor_positions, receptor_mask = receptor2pdbfeature[name_of_receptor]
    
    feature_list = []
    for pdbqt_file in pdbqt_file_list:
        ligand_atom_idx, ligand_positions, ligand_mask = pdbqtvina2feature(pdbqt_file)
        
        # 关键检查：如果配体解析失败（返回空张量），则跳过该配体
        if ligand_atom_idx.numel() == 0:
            continue
            
        atom_idx = torch.cat([receptor_atom_idx, ligand_atom_idx], dim=1)
        positions = torch.cat([receptor_positions, ligand_positions], dim=0)
        positions = torch.unsqueeze(positions, 0)
        mask = torch.cat([receptor_mask, ligand_mask], dim=1)
        feature_list.append((atom_idx, positions, mask))
        
    return feature_list

class ENN(nn.Module):
    """Args:
            1. amino's categories
            2. amino's position
    """
    def __init__(self, latent_dim = 50, device = torch.device('cpu'), is_one_hot = True, layer = 1, vocab_size=6, coordinate_dim = 3):
        super(ENN, self).__init__()
        self.latent_dim = latent_dim
        self.layer = layer
        self.is_one_hot = is_one_hot
        self.vocab_size = vocab_size
        self.coordinate_dim = 3
        self.device = device
        self.aggregate = torch.mean
        if is_one_hot:
            self.node_embedding = nn.Embedding(vocab_size, latent_dim).to(device)

        self.phi_e = nn.Sequential(
                                 nn.Linear(2*self.latent_dim+1, self.latent_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.latent_dim, self.latent_dim),
                                 nn.Tanh()).to(device) ## 2d+1 -> d

        self.phi_x = nn.Sequential(
                                 nn.Linear(self.latent_dim, self.latent_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.latent_dim, 1),
                                 nn.Tanh(),
                                 ).to(device)  ### d->1

        self.phi_h = nn.Sequential(
                                 nn.Linear(self.latent_dim, self.latent_dim),
                                 nn.Tanh(),
                                 nn.Linear(self.latent_dim, self.latent_dim),
                                 nn.Tanh(),
                                 ).to(device) ### d->d

        self.phi_inf = nn.Sequential(
                                 nn.Linear(self.latent_dim, 1),
                                 nn.Sigmoid(),
                                 ).to(device)  ## d->1

        self.output_mlp = nn.Linear(self.latent_dim, 1)

    def forward(self, input_data, coordinate, mask):
        """
        Args:
            input_data: LongTensor(b,N) & FloatTensor(b,N,d)
            coordinate: b,N,3
            mask: b,N
            where b = batchsize, N = max_num_of_atom

        Returns:
            (b,1)

        """
        transform = False
        H = self.node_embedding(input_data) if self.is_one_hot else input_data
        if H.dim() == 4:  ##### (a1,a2,N,d)
            transform = True
            a1,a2,a3,a4 = H.shape
            H = H.view(-1,a3,a4) ## b,N,d
            b1,b2,b3,b4 = coordinate.shape
            coordinate = coordinate.view(-1,b3,b4) ### b,N,3
            d1,d2,d3 = mask.shape
            mask = mask.view(-1,d3)  ## b,N

        b, N = H.shape[0], H.shape[1]
        X = coordinate ### b,N,3
        mask_expand = mask.unsqueeze(-1) #### b,N,1
        mask_expand2 = mask_expand.permute(0,2,1) ### b,1,N
        mask_square = mask_expand * mask_expand2 ### b,N,N
        mask_square = mask_square.unsqueeze(-1) ### b,N,N,1

        for l in range(self.layer):
            ### 1. m_ij = phi_e(h_i, h_j, ||x_i^l - x_j^l||^2)
            H1 = H.unsqueeze(2).repeat(1,1,N,1) ### b,N,N,d
            H2 = H.unsqueeze(1).repeat(1,N,1,1) ### b,N,N,d
            x1 = X.unsqueeze(2).repeat(1,1,N,1) ### b,N,N,3
            x2 = X.unsqueeze(1).repeat(1,N,1,1) ### b,N,N,3
            x12 = torch.sum((x1-x2)**2 * mask_square, dim=-1, keepdim=True) ### b,N,N,1
            H12x = torch.cat([H1,H2,x12], -1) ### b,N,N,2d+1
            M = self.phi_e(H12x)*mask_square ### b,N,N,d
            ### 2. e_ij = phi_inf(m_ij)
            E = self.phi_inf(M) ### b,N,N,1
            ### 3. m_i = \sum e_ij m_ij
            M2 = torch.sum(M*E,1) ## b,N,d
            ### 4. x_i^{l+1} = x_i^l + \sum_{j\neq i} (x_i^l - x_j^l) phi_x(m_ij)
            X = X + torch.sum((x1 - x2) * mask_square * self.phi_x(M), dim=1) ## b,N,3
            ### 5. h_i^{l+1} = phi_h(h_i^l, m_i)
            H = self.phi_h(M2) + H   ### b,N,d
            H = H * mask_expand  ### b,N,d

        if transform:
            H = H.view(a1,a2,a3,a4)
            mask = mask.view(d1,d2,d3)
        H = self.aggregate(H*mask.unsqueeze(-1), dim = -2)
        H = nn.ReLU()(H)
        H = self.output_mlp(H)
        return H

    def forward_ligand_list(self, name_of_receptor, pdbqtvina_list):
        feature_list = featurize_receptor_and_ligand_list(name_of_receptor, pdbqtvina_list)
        
        # 如果特征列表为空（所有配体都解析失败），返回空结果
        if not feature_list:
            return None, []
            
        output_list = []
        for feature_tuple in feature_list:
            if feature_tuple is None:
                continue
            atom_idx, positions, mask = feature_tuple
            output = self.forward(atom_idx, positions, mask) #### [1,1]
            output_list.append(output)

        # 如果所有配体都无效，也返回空结果
        if not output_list:
             return None, []

        outputs = torch.cat(output_list, dim=0).view(-1)
        log_output = F.log_softmax(outputs, 0)
        prob_output = F.softmax(outputs, 0)
        return log_output, prob_output.tolist()

if __name__ == "__main__":
    pass