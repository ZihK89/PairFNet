'''
PairNetOpsV3: network.py
If you are a user of CFS3 at the UoM, we recommend using: "module load apps/binapps/pytorch/2.3.0-311-gpu-cu121" to setup environment
'''
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import read_input, analysis, write_output
import time
import os, pdb

#todo when V3, move to another script   
class Molecule_set(object):
    '''
    This class is to define a molecule for data pre-process, not for training and testing dataset
    Note that previously this class was named molecule!!!
    Old variable name may be misleading: this object is not a traversable object.
    Each object contains the characteristics of each molecule. 
    This object stores the features of all molecules, and each feature is traversable.
    '''
    def __init__(self, total_size, data_dir='./ml_data'):
        '''
        data_dir: the dataset path, the default value is ./ml_data
        energies, coords, forces, charges are NDarray data, convert them into tensor before training
        total_size: training size + val size + testing size
        '''
        self.data_dir = data_dir
        self.total_size = total_size
        self.atoms = []
        self.atom_names = [] 
        #read nuclear charges from input file, and convert into atom name
        element = {1: "H", 6: "C", 7: "N", 8: "O"}
        input_nuclear_charges = open(f"./nuclear_charges.txt", "r")
        input_nuclear_charges = open(f"./nuclear_charges.txt", "r")
        for atom in input_nuclear_charges:
            self.atoms.append(int(atom))
            self.atom_names.append(element[self.atoms[-1]])
        self.n_atom = len(self.atoms)
        
        # read energy from input txt
        self.energies = np.reshape(np.loadtxt(f"./{data_dir}/energies.txt", max_rows = total_size), (total_size))
        if len(self.energies) < total_size:
                print("ERROR - requested set size exceeds the dataset size")
                exit()
        length_check = np.loadtxt(f"./{data_dir}/coords.txt")
        if (length_check.shape[0]%self.n_atom) != 0:
            print("ERROR - mismatch between molecule size and dataset size.")
            print("Check the nuclear_charges.txt file.")
            exit()
        # read coords, force, partial charges from input file
        self.coords = np.reshape(np.loadtxt(f"./{data_dir}/coords.txt",
                max_rows=total_size*self.n_atom), (total_size, self.n_atom, 3))
        self.forces = np.reshape(np.loadtxt(f"./{data_dir}/forces.txt",
                max_rows=total_size*self.n_atom), (total_size, self.n_atom, 3))
        self.charges = np.reshape(np.loadtxt(f"./{data_dir}/charges.txt",
                max_rows=total_size*self.n_atom), (total_size, self.n_atom))
        
        # calculate electrostatic energy
        elec = np.zeros(total_size)
        for s in range(total_size):
            for i in range(len(self.atoms)):
                for j in range(i): 
                    r_ij = np.linalg.norm(self.coords[s][i] - self.coords[s][j])
                    coul_sum = self.charges[s][i] * self.charges[s][j] / r_ij
                    elec[s] = elec[s] + coul_sum
        # kj to kcal
        elec = (elec*(1.0e10)*(6.022e23)*(1.602e-19)**2)/4.184/1000/8.854e-12
        self.elec_energies = elec

class Dataset(torch.utils.data.Dataset):
    def __init__(self, atoms, coords, forces, charges,total_size, energies):

        self.atoms = atoms
        self.coords = coords
        self.forces = forces
        self.charges = charges
        self.size = total_size
        self.energy = energies
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # input features
        charges = torch.tensor(self.atoms, dtype=torch.float32)
        coordinates = torch.tensor(self.coords[idx], dtype=torch.float32)
        # output features
        forces = torch.tensor(self.forces[idx], dtype=torch.float32)
        energy = torch.tensor(self.energy[idx], dtype=torch.float32)
        partial_charges = torch.tensor(self.charges[idx], dtype=torch.float32)
        
        return {
            'input': {'charges': charges, 'coordinates': coordinates},
            'values': {'forces': forces, 'energy': energy, 'partial_charges': partial_charges}
        }
    
class NuclearChargePairs(nn.Module):
    def __init__(self, n_pairs, n_atoms,**kwargs):
        super(NuclearChargePairs, self).__init__()
        self.n_pairs = n_pairs
        self.n_atoms = n_atoms
    
    def forward(self, atom_nc):
        #atom_nc is the nuclear charge for each atom
        a = atom_nc.unsqueeze(2)
        b = atom_nc.unsqueeze(1)
        c = a*b
        tri1 = torch.tril(c)  
        tri2 = torch.diag_embed(torch.diagonal(c, dim1=1, dim2=2)) 
        tri = tri1 - tri2
        
        atom_nc_nonzero_values = tri[tri != 0].view(atom_nc.size(0), self.n_pairs)
        return atom_nc_nonzero_values #[batch size, n_pairs]

class CoordsToNRF(nn.Module):
    def __init__(self, max_nrf,n_pairs,n_atoms,**kwargs):
        super(CoordsToNRF, self).__init__()
        self.max_nrf = max_nrf
        self.n_pairs = n_pairs
        self.n_atoms = n_atoms
        self.au2kcalmola = 627.5095 * 0.529177
    
    def forward(self, coords, atom_nc):# atom_nc: atom nuclear charge
        a = coords.unsqueeze(2)  # (batch_size, n_atoms, 1, 3)
        b = coords.unsqueeze(1)  # (batch_size, 1, n_atoms, 3)
        diff = a - b
        
        diff2 = torch.sum(diff**2, dim=-1)
        tri = torch.tril(diff2, diagonal=-1)
        nonzero_values = tri[tri != 0].view(coords.size(0), -1)
        
        r = torch.sqrt(nonzero_values) 
        recip_r2 = 1 / r**2
        nrf = ((atom_nc * self.au2kcalmola) * recip_r2) / self.max_nrf  # (batch_size, n_pairs)
        nrf = nrf.view(coords.size(0), self.n_pairs)
        
        return nrf # (batch_size, n_pairs)


    
class Eij_layer(nn.Module):
    def __init__(self,n_pairs, max_Eij, **kwargs):
        super(Eij_layer, self).__init__()
        self.n_pairs = n_pairs
        self.max_Eij = max_Eij
    
    def forward(self, decomp_scaled):
        decomp_scaled = decomp_scaled.view(decomp_scaled.size(0), -1)  # (batch_size, -1)
        decomp = (decomp_scaled - 0.5) * (2 * self.max_Eij)
        decomp = decomp.view(decomp_scaled.size(0), self.n_pairs) #(batch_size, n_pairs)
        return decomp

class ERecomposition(nn.Module):
    def __init__(self, n_atoms, n_pairs, **kwargs):
        super(ERecomposition, self).__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_pairs
    
    def forward(self, coords, decompFE):
        coords.requires_grad_(True)
        decompFE = decompFE.view(decompFE.size(0),-1) # (batch_size, n_pairs)
                
        a = coords.unsqueeze(2)
        b = coords.unsqueeze(1)
        
        diff = a-b
        diff2 = torch.sum(diff**2, dim=-1) # (batch_size, n_atom, n_atom)
        tri = torch.tril(diff2, diagonal=-1) # (batch_size, n_atom, n_atom)
        nonzero_values = tri[tri != 0].view(coords.size(0), -1)
        diff_flat = nonzero_values.view(tri.size(0), -1) 
        r_flat = diff_flat**0.5
        recip_r_flat = 1 / r_flat
        norm_recip_r = (recip_r_flat**2).sum(dim=1, keepdim=True)**0.5
        eij_E = recip_r_flat / norm_recip_r

        recompE = torch.einsum('bi, bi -> b', eij_E, decompFE)
        recompE = recompE.view(coords.size(0), 1) #batch_size, 1
        return recompE

class E_layer(nn.Module):
    def __init__(self, prescale, norm_scheme,**kwargs):
        super(E_layer, self).__init__()
        self.prescale = prescale
        self.norm_sheme = norm_scheme
    
    def forward(self, E_scaled):
        if self.norm_sheme == 'z-score':
            E = E_scaled * self.prescale[1] + self.prescale[0]
        elif self.norm_sheme == 'force':
            E = ((E_scaled - self.prescale[2]) /
                (self.prescale[3] - self.prescale[2]) *
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])
        elif self.norm_sheme == 'none':
            E = E_scaled
        E = E.view(E_scaled.size(0), -1)
        return E

class F_layer(nn.Module):
    def __init__(self, n_atoms, n_pairs):
        super(F_layer, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = n_pairs
    
    def forward(self, E, coords):
        # when batch size > 1, E is not a scalar, but a tensor [batch size, 1]
        coords.requires_grad_(True)
        forces = torch.autograd.grad(
                outputs=E, inputs=coords, grad_outputs=torch.ones_like(E),create_graph=True)[0]
        forces = forces*-1
                # retain_graph=True
        # forces = torch.tensor(forces)


        return forces

class Q_layer(nn.Module):
    def __init__(self, n_atoms, n_pairs, charge_scheme):
        super(Q_layer, self).__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_pairs
        self.charge_scheme = charge_scheme
    
    def forward(self, old_q):
        if self.charge_scheme == 1:  
            new_q = old_q
        elif self.charge_scheme == 2:  
            sum_q = torch.sum(old_q, dim=1, keepdim=True) / self.n_atoms
            new_q = old_q - sum_q  
        return new_q

class PairNet(nn.Module):
    '''define the PairNet model
    input: atom_nc, coords
    output: force, energy, partial charge
    '''
    def __init__(self,n_atoms,n_nodes, n_layers, charge_scheme, norm_scheme, prescale):
        # parameters``
        super(PairNet,self).__init__()
        self.n_pairs = int(n_atoms * (n_atoms - 1) / 2)
        self.prescale = torch.tensor(prescale, dtype=torch.float32)
        self.max_NRF = torch.tensor(prescale[4], dtype=torch.float32)
        self.max_matFE = torch.tensor(prescale[5], dtype=torch.float32)
        self.ncp_layer = NuclearChargePairs(n_pairs = self.n_pairs, n_atoms = n_atoms)
        self.coordtoNRF= CoordsToNRF(max_nrf=self.max_NRF,n_pairs=self.n_pairs,n_atoms= n_atoms)
        
        #hidden layers
        hidden_layers = []
        for i in range(n_layers):
            if i == 0:
                hidden_layers.append(nn.Linear(self.n_pairs, n_nodes[i]))
            else: 
                hidden_layers.append(nn.Linear(n_nodes[i-1], n_nodes[i]))
            hidden_layers.append(nn.SiLU())
        self.connect = nn.Sequential(*hidden_layers)
        
        # for interatomic pairwise energy components
        self.output1 = nn.Linear(n_nodes[-1], self.n_pairs)
        self.out1_linear_activ = nn.Linear(self.n_pairs, self.n_pairs)
        #output layer for uncorrected predicted charges
        self.output2 = nn.Linear(n_nodes[-1], n_atoms)
        self.out2_linear_activ = nn.Linear(n_atoms, n_atoms)

        self.unscale_E_layer = Eij_layer(self.n_pairs, self.max_matFE)

        self.ERecomposition = ERecomposition(n_atoms,self.n_pairs)
        self.E_layer = E_layer(prescale, norm_scheme)
        self.F_layer = F_layer(n_atoms, self.n_pairs)
        self.Q_layer = Q_layer(n_atoms,self.n_pairs,charge_scheme)
        
    def forward(self, atom_nc, coords):   
        # coords and z_types to NRFs
        x = self.ncp_layer(atom_nc)
        x = self.coordtoNRF(coords,x)
        x = self.connect(x)
        # output for interatomic pairwise energy components
        out1 = self.output1(x)
        out1 = self.out1_linear_activ(out1)
        # output for uncorrected predicted charges
        out2 = self.output2(x)
        out2 = self.out2_linear_activ(out2)
        # calculated unscaled interatomic energies
        unscale_E = self.unscale_E_layer(out1)
        # calculate the scaled energy from the coordinates and unscaled qFE
        energy = self.ERecomposition(coords, unscale_E)
        # calculate the unscaled energy
        energy = self.E_layer(energy)
        # energy = energy.squeeze(1)
        # obtain the forces by taking the gradient of the energy
        force = self.F_layer(energy, coords)
        # predict partial charges
        charge = self.Q_layer(out2)
                
        return force, energy, charge
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def train(model,train_loader, train_size,optimizer,device,loss_weights,loss_fn,n_atoms):
    """
    Use this function to train the model
    """
    model.to(device)
    model.train()
    running_loss = 0
    running_loss_f = 0
    running_loss_e = 0
    running_loss_q = 0
    
    for data in train_loader:
        atom_nc = data['input']['charges'].to(device)
        coords = data['input']['coordinates'].to(device)
        target_force = data['values']['forces'].to(device)
        target_energy = data['values']['energy'].to(device)
        target_charge = data['values']['partial_charges'].to(device)
        
        # forward
        outputs = model(atom_nc, coords)
        # loss
        optimizer.zero_grad()
        force_pred, energy_pred, charge_pred = outputs
        loss_energy = loss_fn(energy_pred, target_energy)
        loss_force = loss_fn(force_pred, target_force)
        loss_charge = loss_fn(charge_pred, target_charge)
        # loss_energy = loss_energy / (3*n_atoms)
        
        total_loss = loss_weights[0]*loss_force+loss_weights[1]*loss_energy+loss_weights[2]*loss_charge
        
        # backward
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss
        running_loss_f += loss_force
        running_loss_e += loss_energy
        running_loss_q += loss_charge
    
    loss_mean = running_loss / len(train_loader)
    loss_f = running_loss_f / len(train_loader)
    loss_e = running_loss_e / len(train_loader)
    loss_q = running_loss_q / len(train_loader)
    return [loss_mean, loss_f, loss_e, loss_q]

def val(model, val_loader,val_size, device, loss_weights, loss_fn,n_atoms):
    '''
    Use this function to validate model
    '''
    model.to(device)
    model.eval()
    running_loss = 0
    running_loss_f = 0
    running_loss_e = 0
    running_loss_q = 0
    
    for data in val_loader:
        atom_nc = data['input']['charges'].to(device)
        coords = data['input']['coordinates'].to(device)
        target_force = data['values']['forces'].to(device)
        target_energy = data['values']['energy'].to(device)
        target_charge = data['values']['partial_charges'].to(device)

        outputs = model(atom_nc, coords)
        force_pred,energy_pred,charge_pred = outputs
            
        loss_energy = loss_fn(energy_pred, target_energy)
        loss_force = loss_fn(force_pred, target_force)
        loss_charge = loss_fn(charge_pred, target_charge)
        # loss_energy = loss_energy / (3*n_atoms)

        total_loss = loss_weights[0]*loss_force+loss_weights[1]*loss_energy+loss_weights[2]*loss_charge


        running_loss += total_loss
        running_loss_f += loss_force
        running_loss_e += loss_energy
        running_loss_q += loss_charge
            
    loss_mean = running_loss/len(val_loader)
    loss_f = running_loss_f/len(val_loader)
    loss_e = running_loss_e/len(val_loader)
    loss_q = running_loss_q/len(val_loader)

    return [loss_mean, loss_f, loss_e, loss_q]
    
def test(model,test_loader,device,test_size,loss_fn):
    '''
    Use this function to test the model
    '''
    f_pred_list = []
    e_pred_list = []
    q_pred_list = []
    
    model.to(device)
    model.eval()
    for data in test_loader:
        atom_nc = data['input']['charges'].to(device)
        coords = data['input']['coordinates'].to(device)
        target_force = data['output']['forces'].to(device)
        target_energy = data['output']['energy'].to(device)
        target_charge = data['output']['partial_charges'].to(device)
        # forward
        outputs = model(atom_nc, coords)
        force_pred, energy_pred, charge_pred = outputs
        f_pred_list.append(force_pred)
        e_pred_list.append(energy_pred)
        q_pred_list.append(charge_pred)
    return f_pred_list, e_pred_list, q_pred_list


def summary(test_set,y_hat,test_size, output_dir,lable):
    mean_ae = 0
    max_ae = 0
    for actual, prediction in zip(test_set, y_hat):
        diff = prediction - actual
        mean_ae += np.sum(abs(diff))
        if abs(diff) > max_ae:
            max_ae = abs(diff)
    mean_ae = mean_ae / test_size
    L = write_output.scurve(test_set.flatten(), y_hat.flatten(),
                            output_dir, f"{lable}_scurve", val)
    
    return mean_ae, max_ae, L
        
    
def main():
    """
    read model parameters and ML dataset -> define a Molecule object 
    -> define a Dataset class -> pre-process the data ->
    build the model -> train and test
    """
    # read PairNet model parameters
    ann_params = read_input.ann("ann_params.txt")
    n_data = ann_params["n_data"]
    n_train, n_val, n_test = n_data[0], n_data[1], n_data[2]
    n_nodes = ann_params["n_nodes"]
    n_layers = ann_params["n_layers"]
    charge_scheme = ann_params["charge_scheme"]
    epochs = ann_params["epochs"]
    init_lr = ann_params["init_lr"]
    min_lr = ann_params["min_lr"]
    lr_patience = ann_params["lr_patience"]
    lr_factor = ann_params["lr_factor"]
    batch_size = ann_params["batch_size"]
    loss_weights = ann_params["loss_weights"]
    size = n_train + n_val + n_test
    output_dir1 = "./plots_and_data"
    isExist = os.path.exists(output_dir1)
    if not isExist:
        os.makedirs(output_dir1)
    
    # define a Mol object
    mol_set = Molecule_set(size)
    n_atoms = mol_set.n_atom    
    # pre-process the data
    norm_scheme = ann_params["norm_scheme"]
    mol_set.orig_energies = np.copy(mol_set.energies)
    mol_set.trainval = [*range(0, n_train + n_val, 1)]
    print("Calculating pairwise energies...")
    trainval = [*range(0, n_train + n_val, 1)]
    trainval_forces = np.take(mol_set.forces, trainval, axis=0)
    trainval_energies = np.take(mol_set.energies, trainval, axis=0)
    prescale = analysis.prescale_e(mol_set, trainval_energies,
                trainval_forces, norm_scheme)
    analysis.get_eij(mol_set, size, output_dir1)
    prescale = analysis.prescale_eij(mol_set, prescale)
    
    # dataset
    dataset = Dataset(mol_set.atoms,mol_set.coords,mol_set.forces,mol_set.charges,mol_set.total_size,mol_set.energies)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])
    train_dataset = Subset(dataset, list(range(n_train)))
    val_dataset = Subset(dataset, list(range(n_val)))
    test_dataset = Subset(dataset, list(range(n_test)))


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

    # check GPUs
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('GPU detected, will use GPU for training and testing')
    else:
        device = torch.device("cpu")
        print('No GPU detected, will use CPU for training')
    
    # PairNet model
    model = PairNet(n_atoms,n_nodes,n_layers,charge_scheme,norm_scheme,prescale)
    # 
    model.apply(init_weights)
    # loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    # parameter_list = list(model.connect.parameters())
    optimizer = optim.Adam(model.connect.parameters(), lr=init_lr,eps=1e-7,amsgrad=False)

    # train
    print("Training model...")
    for epoch in range(epochs):
        start_time = time.time()  
        train_loss = train(model,train_loader,n_train,optimizer,device,loss_weights,loss_fn,n_atoms)
        val_loss = val(model,val_loader,n_val,device,loss_weights,loss_fn,n_atoms)
        end_time = time.time()
        epoch_time = end_time - start_time
        #! just for testing
        print(f'Epoch: {epoch+1}, time: {epoch_time:.2f}, train loss:{train_loss[0]} val loss: {val_loss[0]}')
    
    # test
    # print('Testing model...')
    # f_pred_list, e_pred_list, q_pred_list = test(model,test_loader,device,n_test,loss_fn)
    
    

    
if __name__ == "__main__":
    main()