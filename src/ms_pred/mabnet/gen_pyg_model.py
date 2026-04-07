"""DAG Gen model EGT version"""
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
# import dgl
# import dgl.nn as dgl_nn


import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as magma
import ms_pred.mabnet.dag_pyg_data as dag_data

from torch_geometric.nn import global_mean_pool, GlobalAttention
from ms_pred.mabnet.egt import egt_model
from .utils import dgl_to_pyg
from torch_geometric.data import Data, Batch

class FragEGT(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        set_layers: int = 2,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        dropout: float = 0,
        mpnn_type: str = "GGNN",
        pool_op: str = "avg",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        pe_embed_k: int = 0,
        max_broken: int = magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        root_encode: str = "gnn",
        inject_early: bool = False,
        warmup: int = 1000,
        embed_adduct: bool = False,
        encode_forms: bool = False,
        add_hs: bool = False,
        mabnet_layers: int = 2,
        mabnet_heads: int = 4,
        many_body: bool = False,
        edge_update: bool = False,
        frag_layers: int = 8,
        frag_heads: int = 4,
        input_node_dim: int = 48,
        embed_instrumentation: bool = False,
        **kwargs,
    ):
        """__init__ _summary_"""
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.root_encode = root_encode
        self.pe_embed_k = pe_embed_k
        self.embed_adduct = embed_adduct
        self.encode_forms = encode_forms
        self.add_hs = add_hs

        self.tree_processor = dag_data.TreeProcessor(
            root_encode=root_encode, pe_embed_k=pe_embed_k, add_hs=self.add_hs
        )
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim

            # Account for diffs
            self.formula_in_dim *= 2

        self.pool_op = pool_op
        self.inject_early = inject_early

        self.layers = layers
        self.mpnn_type = mpnn_type
        self.set_layers = set_layers

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = torch.nn.Parameter(torch.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        self.embed_instrumentation = embed_instrumentation  # 控制是否启用instrument

        # NEW: instrument one-hot嵌入矩阵
        if self.embed_instrumentation:
            num_instrument_types = len(common.CANOPUS_INSTRUMENTATION_ONEHOT)
            self.instrument_embedder = nn.Parameter(torch.eye(num_instrument_types).float())
            self.instrument_embedder.requires_grad = False  # one-hot不需要训练

        edge_feats = fragmentation.MAX_BONDS

        orig_node_feats = node_feats
        if self.inject_early:
            node_feats = node_feats + self.hidden_size

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        # MODIFIED: 如果启用instrumentation，调整input_node_dim以包含instrument维度
        # if self.embed_instrumentation:
        #     input_node_dim += num_instrument_types  # 增加one-hot维度

        # Define network
        self.gnn = egt_model.EdgeEnhancedGraphTransformer2D(
            input_node_dim=input_node_dim,
            hidden_dim=hidden_size,
            num_heads=mabnet_heads,
            num_layers=mabnet_layers,
            edge_update=edge_update,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn

            # if inject early, need separate root and child GNN's
            if self.inject_early:
                self.root_module = egt_model.EdgeEnhancedGraphTransformer2D(
                    input_node_dim=input_node_dim,
                    hidden_dim=hidden_size,
                    num_heads=mabnet_heads,
                    num_layers=mabnet_layers,
                    edge_update=edge_update,
                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                use_residuals=True,
                num_layers=1,
            )
        else:
            raise ValueError()

        # MLP layer to take representations from the pooling layer
        # And predict a single scalar value at each of them
        # I.e., Go from size B x 2h -> B x 1
        # MODIFIED: 如果启用instrumentation，调整MLP输入维度
        mlp_input_size = self.hidden_size * 3 + self.max_broken + self.formula_in_dim
        if self.embed_instrumentation:
            mlp_input_size += num_instrument_types
        self.output_map = nn_utils.MLPBlocks(
            input_size=mlp_input_size,
            hidden_size=self.hidden_size,
            output_size=1,
            dropout=self.dropout,
            num_layers=1,
            use_residuals=True,
        )

        if self.pool_op == "avg":
            self.pool = global_mean_pool
        elif self.pool_op == "attn":
            self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        broken,
        adducts,
        instruments=None,  # MODIFIED: instruments现在是必需参数，如果instrumentation=True
        root_forms=None,
        frag_forms=None,
    ):
        """forward _summary_ (略，保持原样)"""
        embed_adducts = self.adduct_embedder[adducts.long()]

        # NEW: 处理instrument one-hot编码（如果启用）
        embed_instruments = None
        if self.embed_instrumentation:
            if instruments is None:
                raise ValueError("instruments must be provided when instrumentation is True")
            embed_instruments = self.instrument_embedder[instruments.long()]  # (B, num_instrument_types)

        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            root_repr_clone = root_repr.clone()
            root_batch_sizes = torch.bincount(root_repr_clone.batch) if root_repr_clone.batch is not None else torch.tensor([root_repr_clone.num_nodes])
            embed_adducts_expand = embed_adducts.repeat_interleave(root_batch_sizes, 0)  # (num_nodes, 10)
            root_h = torch.cat([root_repr_clone.h, embed_adducts_expand], -1)

            # NEW: 如果启用，注入instrument到根节点特征
            if self.embed_instrumentation:
                embed_instruments_expand = embed_instruments.repeat_interleave(root_batch_sizes, 0)
                root_h = torch.cat([root_h, embed_instruments_expand], -1)

            root_repr_clone.h = root_h
            _, root_embeddings = self.root_module(root_repr_clone)
        else:
            pass

        # Line up the features to be parallel between fragment avgs and root
        # graphs
        ext_root = root_embeddings[ind_maps]  # (n_frags, hidden_size)
        graph_batch_sizes = torch.bincount(graphs.batch) if graphs.batch is not None else torch.tensor([graphs.num_nodes])
        ext_root_atoms = torch.repeat_interleave(ext_root, graph_batch_sizes, dim=0) 
        concat_list = [graphs.h]

        if self.inject_early:
            concat_list.append(ext_root_atoms)

        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]    # (n_frag, 10)
            adducts_exp = torch.repeat_interleave(      # (f_nodes, 10)
                adducts_mapped, graph_batch_sizes, dim=0
            )
            concat_list.append(adducts_exp) # [graph.h, adducts]

        # NEW: 注入instrument到碎片节点特征（推荐位置，与adduct类似）
        if self.embed_instrumentation:
            instruments_mapped = embed_instruments[ind_maps]  # (n_frags, num_instrument_types)
            instruments_exp = torch.repeat_interleave(
                instruments_mapped, graph_batch_sizes, dim=0
            )  # (f_nodes, num_instrument_types)
            concat_list.append(instruments_exp)

        graphs.h = torch.cat(concat_list, -1).float()

        frag_embeddings, avg_frags = self.gnn(graphs)  # (f_nodes, 512)

        # Average embed the full root molecules and fragments
        # avg_frags = self.pool(graphs, frag_embeddings)  # (n_frag, 512)

        # Extend the avg of each fragment
        ext_frag_atoms = torch.repeat_interleave(   # (f_nodes, 512)
            avg_frags, graph_batch_sizes, dim=0
        )

        exp_num = graph_batch_sizes  # (n_frag,)
        # Do the same with the avg fragments

        broken = torch.clamp(broken, max=self.broken_clamp) #(f_nodes,)
        ext_frag_broken = torch.repeat_interleave(broken, exp_num, dim=0)   # (n_frag,)
        broken_onehots = self.broken_onehot[ext_frag_broken.long()] # (f_nodes, self.broken_clamp)

        mlp_cat_vec = [
            ext_root_atoms,
            ext_root_atoms - ext_frag_atoms,
            frag_embeddings,
            broken_onehots, #[fnodes, self.max_broken]
        ]
        if self.encode_forms:
            root_exp = root_forms[ind_maps] # (n_frags, 18)
            diffs = root_exp - frag_forms
            form_encodings = self.embedder(frag_forms)  # (n_frags, 144)
            diff_encodings = self.embedder(diffs)   # (n_frags, 144)
            form_atom_exp = torch.repeat_interleave(form_encodings, exp_num, dim=0) # (f_nodes, 144)
            diff_atom_exp = torch.repeat_interleave(diff_encodings, exp_num, dim=0) # (f_nodes, 144)

            mlp_cat_vec.extend([form_atom_exp, diff_atom_exp])

        # NEW: 可选 - 如果想注入到MLP中（而非节点特征），可以在这里cat
        if self.embed_instrumentation:
            instruments_exp_mlp = torch.repeat_interleave(instruments_mapped, exp_num, dim=0)
            mlp_cat_vec.append(instruments_exp_mlp)

        hidden = torch.cat( # (f_nodes, 1831 + num_instrument_types if injected)
            mlp_cat_vec,
            dim=1,
        )

        output = self.output_map(hidden)    # (f_nodes, 1)
        output = self.sigmoid(output)    # (f_nodes, 1)
        padded_out = nn_utils.pad_packed_tensor(output, graph_batch_sizes, 0)    # (n_frags, max(nodes), 1)
        padded_out = torch.squeeze(padded_out, -1)  # (n_frags, max(nodes))
        return padded_out

    def loss_fn(self, outputs, targets, natoms):
        """loss_fn.

        Args:
            outputs: Outputs after sigmoid fucntion
            targets: Target binary vals
            natoms: Number of atoms in each atom to consider padding

        """
        loss = self.bce_loss(outputs, targets.float())
        is_valid = (
            torch.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
        )
        pooled_loss = torch.sum(loss * is_valid) / torch.sum(natoms)
        return pooled_loss

    def _common_step(self, batch, name="train"):
        pred_leaving = self.forward(
            batch["frag_graphs"],
            batch["root_reprs"],
            batch["inds"],
            broken=batch["broken_bonds"],
            adducts=batch["adducts"],
            instruments=batch["instrumentations"] if self.embed_instrumentation else None,
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
        )
        loss = self.loss_fn(pred_leaving, batch["targ_atoms"], batch["frag_atoms"])
        self.log(
            f"{name}_loss", loss.item(), on_epoch=True, batch_size=len(batch["names"])
        )
        return loss

    def training_step(self, batch, batch_idx):
        """training_step."""
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        """validation_step."""
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        """test_step."""
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        """configure_optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }
        return ret

    def predict_mol(
        self,
        root_smi: str,
        adduct: str,
        instrument: str = None,
        threshold: float = 0,
        device: str = "cpu",
        max_nodes: int = None,
    ) -> dict:
        """Predict a new fragmentation tree from a starting root molecule autoregressively.

        Args:
            root_smi (str): SMILES string of the root molecule.
            adduct (str): Adduct type.
            threshold (float): Leaving probability threshold. Default is 0.
            device (str): Device to run computations on ('cpu' or 'cuda'). Default is 'cpu'.
            max_nodes (int, optional): Maximum number of nodes to include. Default is None.

        Returns:
            dict: Dictionary containing fragmentation results.
        """
        # Step 1: Get a fragmentation engine for root molecule
        engine = fragmentation.FragmentEngine(root_smi)
        max_depth = engine.max_tree_depth
        root_frag = engine.get_root_frag()
        root_form = common.form_from_smi(root_smi)
        root_form_vec = torch.FloatTensor(common.formula_to_dense(root_form)).reshape(1, -1).to(device)
        adducts = torch.LongTensor([common.ion2onehot_pos[adduct]]).to(device)
        instruments = torch.LongTensor([common.CANOPUS_INSTRUMENTATION_ONEHOT.get(instrument, 0)]).to(device)
        # Step 2: Featurize the root molecule
        root_graph_dict = self.tree_processor.featurize_frag(
            frag=root_frag,
            engine=engine,
            add_random_walk=True,
        )
        # print(root_graph_dict)
        # print(f'self.root_encode: {self.root_encode}')
        root_repr = None
        if self.root_encode == "gnn":
            # Convert DGL graph to PyG Data object
            root_repr = dgl_to_pyg(root_graph_dict["graph"]).to(device)
        elif self.root_encode == "fp":
            root_fp = torch.from_numpy(np.array(common.get_morgan_fp_smi(root_smi)))
            root_repr = root_fp.float().to(device)[None, :]
        else:
            raise ValueError(f"Unsupported root_encode: {self.root_encode}")

        form_to_min_score = {}
        frag_hash_to_entry = {}
        frag_to_hash = {}
        stack = [root_frag]
        depth = 0
        root_hash = engine.wl_hash(root_frag)
        frag_to_hash[root_frag] = root_hash
        root_score = engine.score_fragment(root_frag)[1]
        id_ = 0
        root_entry = {
            "frag": int(root_frag),
            "frag_hash": root_hash,
            "parents": [],
            "atoms_pulled": [],
            "left_pred": [],
            "max_broken": 0,
            "tree_depth": 0,
            "id": 0,
            "prob_gen": 1,
            "score": root_score,
        }
        id_ += 1
        root_entry.update(engine.atom_pass_stats(root_frag, depth=0))
        form_to_min_score[root_entry["form"]] = root_entry["score"]
        frag_hash_to_entry[root_hash] = root_entry

        # Step 3: Run the autoregressive generation loop
        with torch.no_grad():
            while len(stack) > 0 and depth < max_depth:
                # Convert all new fragments to PyG Data objects
                new_dgl_dicts = [
                    self.tree_processor.featurize_frag(
                        frag=i, engine=engine, add_random_walk=True
                    )
                    for i in stack
                ]
                # print(f"new_dgl_dicts: {new_dgl_dicts[0]['graph']}")
                # Filter out fragments with fewer than 2 nodes
                tuple_list = [
                    (i, j)
                    for i, j in zip(new_dgl_dicts, stack)
                    if i["graph"].num_nodes() > 1
                ]

                if len(tuple_list) == 0:
                    break

                new_dgl_dicts, stack = zip(*tuple_list)
                # Convert DGL graphs to PyG Data objects
                mol_batch_graph = [dgl_to_pyg(i["graph"]) for i in new_dgl_dicts]
                frag_forms = [i["form"] for i in new_dgl_dicts]
                frag_form_vecs = torch.FloatTensor(np.array([common.formula_to_dense(f) for f in frag_forms])).to(device)
                new_frag_hashes = [engine.wl_hash(i) for i in stack]
                frag_to_hash.update(dict(zip(stack, new_frag_hashes)))

                # Batch PyG Data objects
                frag_batch = Batch.from_data_list(mol_batch_graph).to(device)
                inds = torch.zeros(len(mol_batch_graph), dtype=torch.long).to(device)

                broken_nums_ar = np.array([frag_hash_to_entry[i]["max_broken"] for i in new_frag_hashes])
                broken_nums_tensor = torch.FloatTensor(broken_nums_ar).to(device)

                # Forward pass with PyG graphs
                pred_leaving = self.forward(
                    graphs=frag_batch,
                    root_repr=root_repr,
                    ind_maps=inds,
                    broken=broken_nums_tensor,
                    adducts=adducts,
                    instruments=instruments if self.embed_instrumentation else None,
                    root_forms=root_form_vec,
                    frag_forms=frag_form_vecs,
                )
                depth += 1

                # Rank order all atom predictions
                cur_probs = sorted([i["prob_gen"] for i in frag_hash_to_entry.values()])[::-1]
                min_prob = threshold if max_nodes is None or len(cur_probs) < max_nodes else cur_probs[max_nodes - 1]

                new_items = list(zip(stack, new_frag_hashes, pred_leaving, new_dgl_dicts))
                sorted_order = []
                for item_ind, item in enumerate(new_items):
                    frag_hash = item[1]
                    pred_vals_f = item[2]
                    parent_prob = frag_hash_to_entry[frag_hash]["prob_gen"]
                    for atom_ind, (atom_pred, prob_gen) in enumerate(zip(pred_vals_f, parent_prob * pred_vals_f)):
                        sorted_order.append(
                            dict(
                                item_ind=item_ind,
                                atom_ind=atom_ind,
                                prob_gen=prob_gen.item(),
                                atom_pred=atom_pred.item(),
                            )
                        )

                sorted_order = sorted(sorted_order, key=lambda x: -x["prob_gen"])
                new_stack = []

                # Process ordered list
                for new_item in sorted_order:
                    prob_gen = new_item["prob_gen"]
                    atom_ind = new_item["atom_ind"]
                    atom_pred = new_item["atom_pred"]
                    item_ind = new_item["item_ind"]

                    if prob_gen <= min_prob:
                        continue

                    orig_entry = new_items[item_ind]
                    frag_int = orig_entry[0]
                    frag_hash = orig_entry[1]
                    dgl_dict = orig_entry[3]
                    atom = dgl_dict["new_to_old"][atom_ind]
                    out_dicts = engine.remove_atom(frag_int, int(atom))

                    frag_hash_to_entry[frag_hash]["atoms_pulled"].append(int(atom))
                    frag_hash_to_entry[frag_hash]["left_pred"].append(float(atom_pred))
                    parent_broken = frag_hash_to_entry[frag_hash]["max_broken"]

                    for out_dict in out_dicts:
                        out_hash = out_dict["new_hash"]
                        out_frag = out_dict["new_frag"]
                        rm_bond_t = out_dict["rm_bond_t"]
                        frag_to_hash[out_frag] = out_hash
                        current_entry = frag_hash_to_entry.get(out_hash)
                        max_broken = parent_broken + rm_bond_t

                        if current_entry is None:
                            score = engine.score_fragment(int(out_frag))[1]
                            new_stack.append(out_frag)
                            new_entry = {
                                "frag": int(out_frag),
                                "frag_hash": out_hash,
                                "score": score,
                                "id": id_,
                                "parents": [frag_hash],
                                "atoms_pulled": [],
                                "left_pred": [],
                                "max_broken": max_broken,
                                "tree_depth": depth,
                                "prob_gen": prob_gen,
                            }
                            id_ += 1
                            new_entry.update(engine.atom_pass_stats(out_frag, depth=max_broken))
                            temp_form = new_entry["form"]
                            prev_best_score = form_to_min_score.get(temp_form, float("inf"))
                            form_to_min_score[temp_form] = min(new_entry["score"], prev_best_score)
                            frag_hash_to_entry[out_hash] = new_entry
                        else:
                            current_entry["parents"].append(frag_hash)
                            current_entry["prob_gen"] = max(current_entry["prob_gen"], prob_gen)

                        cur_probs = sorted([i["prob_gen"] for i in frag_hash_to_entry.values()])[::-1]
                        min_prob = threshold if max_nodes is None or len(cur_probs) < max_nodes else cur_probs[max_nodes - 1]

                    stack = new_stack

        # Filter fragments to keep only the minimum score for each formula
        frag_hash_to_entry = {
            k: v for k, v in frag_hash_to_entry.items() if form_to_min_score[v["form"]] == v["score"]
        }

        # Apply max_nodes limit if specified
        if max_nodes is not None:
            sorted_keys = sorted(
                list(frag_hash_to_entry.keys()),
                key=lambda x: -frag_hash_to_entry[x]["prob_gen"],
            )
            frag_hash_to_entry = {k: frag_hash_to_entry[k] for k in sorted_keys[:max_nodes]}

        return frag_hash_to_entry