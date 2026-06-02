"""Visualize the GNN's view of a molecule.

Draws the molecular graph as the GNN sees it: atoms as nodes with feature
vectors, bonds as edges with feature vectors, and annotates one round of
message passing to show exactly what happens during a GINEConv forward pass.

Run:  python scripts/visualize_gnn_graph.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from molgate.data.featurizer import (
    _atom_features,
    _bond_features,
    ATOM_TYPES,
    BOND_TYPES,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# --- Settings ---
SMILES = "c1cc(O)ccc1N"  # 4-aminophenol: aromatic ring + OH + NH2
MOL = Chem.MolFromSmiles(SMILES)
# Generate 2D coordinates using RDKit's coordinate generator
AllChem.Compute2DCoords(MOL)

# Human-readable atom type names
ATOM_NAMES = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
BOND_NAMES = {
    Chem.rdchem.BondType.SINGLE: "single",
    Chem.rdchem.BondType.DOUBLE: "double",
    Chem.rdchem.BondType.TRIPLE: "triple",
    Chem.rdchem.BondType.AROMATIC: "arom",
}

# Atom colors by element
ATOM_COLORS = {
    6: "#505050",   # C - dark gray
    7: "#3050F8",   # N - blue
    8: "#FF2020",   # O - red
}


def decode_atom_features(atom):
    """Human-readable summary of an atom's feature vector."""
    feats = _atom_features(atom)
    atom_type_vec = feats[:9]
    atom_type_idx = atom_type_vec.index(1) if 1 in atom_type_vec else -1
    atom_sym = ATOM_NAMES.get(ATOM_TYPES[atom_type_idx], "?") if atom_type_idx >= 0 else "?"
    return {
        "symbol": atom_sym,
        "degree": int(feats[9]),
        "charge": int(feats[10]),
        "num_hs": int(feats[11]),
        "aromatic": bool(feats[12]),
        "in_ring": bool(feats[13]),
        "raw_vector": feats,
    }


def decode_bond_features(bond):
    """Human-readable summary of a bond's feature vector."""
    feats = _bond_features(bond)
    bond_type_vec = feats[:4]
    bond_type_idx = bond_type_vec.index(1) if 1 in bond_type_vec else -1
    bond_name = BOND_NAMES.get(BOND_TYPES[bond_type_idx], "?") if bond_type_idx >= 0 else "?"
    return {
        "type": bond_name,
        "conjugated": bool(feats[4]),
        "in_ring": bool(feats[5]),
        "raw_vector": feats,
    }


def get_rdkit_2d_coords(mol):
    """Extract RDKit 2D coordinates as a dict of {atom_idx: (x, y)}."""
    conf = mol.GetConformer()
    pos = {}
    for i in range(mol.GetNumAtoms()):
        pt = conf.GetAtomPosition(i)
        pos[i] = (pt.x, pt.y)
    return pos


# ===========================================================================
# Collect atom/bond info
# ===========================================================================
atom_info = {}
for atom in MOL.GetAtoms():
    atom_info[atom.GetIdx()] = decode_atom_features(atom)

bond_info = {}
edges = []
for bond in MOL.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bond_info[(i, j)] = decode_bond_features(bond)
    bond_info[(j, i)] = bond_info[(i, j)]  # symmetric lookup
    edges.append((i, j))

pos = get_rdkit_2d_coords(MOL)


# ===========================================================================
# Figure 1: Molecular graph with RDKit layout + message passing detail
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# --- Left panel: molecular graph ---
ax1 = axes[0]
ax1.set_title("Molecular Graph (GNN Input)\n4-aminophenol: c1cc(O)ccc1N",
              fontsize=14, fontweight="bold", pad=15)

# Scale coordinates for nice spacing
coords = np.array([pos[i] for i in range(len(pos))])
coords -= coords.mean(axis=0)
scale = 2.0 / max(coords.max() - coords.min(), 1e-6)
coords *= scale
scaled_pos = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}

# Draw bonds (edges)
for (i, j) in edges:
    xi, yi = scaled_pos[i]
    xj, yj = scaled_pos[j]
    binfo = bond_info[(i, j)]
    # Aromatic bonds get a dashed style
    ls = "--" if binfo["type"] == "arom" else "-"
    lw = 2.5 if binfo["type"] == "arom" else 3.0
    ax1.plot([xi, xj], [yi, yj], color="#999999", linewidth=lw, linestyle=ls, zorder=1)
    # Bond label at midpoint
    mx, my = (xi + xj) / 2, (yi + yj) / 2
    # Offset label perpendicular to bond direction
    dx, dy = xj - xi, yj - yi
    length = max(np.sqrt(dx**2 + dy**2), 1e-6)
    nx_, ny_ = -dy / length, dx / length  # perpendicular
    ax1.text(mx + nx_ * 0.12, my + ny_ * 0.12, binfo["type"],
             fontsize=7, color="#777777", ha="center", va="center", fontstyle="italic")

# Draw atoms (nodes)
node_radius = 0.18
for idx, info in atom_info.items():
    x, y = scaled_pos[idx]
    color = ATOM_COLORS.get(MOL.GetAtomWithIdx(idx).GetAtomicNum(), "#AAAAAA")
    circle = plt.Circle((x, y), node_radius, color=color, ec="black",
                         linewidth=2, zorder=3)
    ax1.add_patch(circle)
    ax1.text(x, y, f"{info['symbol']}{idx}", ha="center", va="center",
             fontsize=11, color="white", fontweight="bold", zorder=4)

    # Feature annotation box below each atom
    parts = [f"deg={info['degree']}", f"Hs={info['num_hs']}"]
    if info["aromatic"]:
        parts.append("arom")
    if info["in_ring"]:
        parts.append("ring")
    feat_str = " ".join(parts)
    ax1.text(x, y - node_radius - 0.10, feat_str, ha="center", va="top",
             fontsize=6.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="#f0f0f0", alpha=0.85, edgecolor="#cccccc"))

# Feature vector legend
legend_text = (
    "Node features (14-dim per atom):\n"
    "  [atom_type₉ | degree₁ | charge₁ | Hs₁ | arom₁ | ring₁]\n\n"
    "Edge features (6-dim per bond):\n"
    "  [bond_type₄ | conjugated₁ | ring₁]"
)
ax1.text(0.02, 0.02, legend_text, transform=ax1.transAxes, fontsize=8,
         fontfamily="monospace", va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffff0", edgecolor="#cccc00", alpha=0.9))

ax1.set_xlim(-2.0, 2.0)
ax1.set_ylim(-2.2, 2.2)
ax1.set_aspect("equal")
ax1.axis("off")

# --- Right panel: message passing detail for atom C2 ---
ax2 = axes[1]
target_atom = 2  # C2: has 3 neighbors (C1, O3, C6) with mixed bond types
neighbors = [n for n in range(MOL.GetNumAtoms())
             if MOL.GetBondBetweenAtoms(target_atom, n) is not None]

ax2.set_title(
    f"Message Passing (1 round) → atom {atom_info[target_atom]['symbol']}{target_atom}\n"
    "How GINEConv uses both atom and bond embeddings",
    fontsize=14, fontweight="bold", pad=15,
)

# Layout: target center, neighbors around it at equal angles
center = np.array([0.0, 0.6])
n_nbrs = len(neighbors)
angles = np.linspace(np.pi * 0.2, np.pi * 0.8, n_nbrs) if n_nbrs > 1 else [np.pi / 2]
nbr_pos = {}
for n, a in zip(neighbors, angles):
    nbr_pos[n] = center + 1.6 * np.array([np.cos(a), np.sin(a)])

# Draw message arrows and annotate the computation per neighbor
for n in neighbors:
    ninfo = atom_info[n]
    nx_, ny_ = nbr_pos[n]
    cx, cy = center

    # Direction vector
    dx, dy = cx - nx_, cy - ny_
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist

    # Arrow from neighbor to target
    ax2.annotate(
        "",
        xy=(cx - ux * 0.22, cy - uy * 0.22),
        xytext=(nx_ + ux * 0.18, ny_ + uy * 0.18),
        arrowprops=dict(arrowstyle="-|>", color="#E06030", lw=2.5, mutation_scale=18),
        zorder=2,
    )

    # Neighbor node
    ncolor = ATOM_COLORS.get(MOL.GetAtomWithIdx(n).GetAtomicNum(), "#AAAAAA")
    circ = plt.Circle((nx_, ny_), 0.16, color=ncolor, ec="black", linewidth=1.5, zorder=5)
    ax2.add_patch(circ)
    ax2.text(nx_, ny_, f"{ninfo['symbol']}{n}", ha="center", va="center",
             fontsize=10, color="white", fontweight="bold", zorder=6)

    # h_u label (atom embedding output)
    ax2.text(nx_, ny_ + 0.25, f"h_{n} ∈ ℝ¹²⁸", ha="center", fontsize=8, color="#0055AA",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="#d0e8ff", alpha=0.9, edgecolor="#0055AA"))

    # Bond feature label on the arrow
    bkey = (min(target_atom, n), max(target_atom, n))
    if bkey in bond_info:
        binfo = bond_info[bkey]
        mid = np.array([(nx_ + cx) / 2, (ny_ + cy) / 2])
        # Perpendicular offset
        perp = np.array([-uy, ux]) * 0.18
        ax2.text(mid[0] + perp[0], mid[1] + perp[1],
                 f"e_{min(target_atom,n)},{max(target_atom,n)} = {binfo['type']}\n∈ ℝ¹²⁸ (after encoding)",
                 fontsize=7, color="#E06030", ha="center", fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="#fff0e0", alpha=0.8, edgecolor="#E06030"))

# Target node (larger)
tcolor = ATOM_COLORS.get(MOL.GetAtomWithIdx(target_atom).GetAtomicNum(), "#AAAAAA")
circ = plt.Circle(center, 0.20, color=tcolor, ec="black", linewidth=2.5, zorder=5)
ax2.add_patch(circ)
ax2.text(center[0], center[1], f"{atom_info[target_atom]['symbol']}{target_atom}",
         ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=6)
ax2.text(center[0], center[1] - 0.30, f"h_{target_atom} ∈ ℝ¹²⁸", ha="center", fontsize=9,
         color="#0055AA", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.15", facecolor="#d0e8ff", alpha=0.9, edgecolor="#0055AA"))

# GINEConv formula — expanded to show where encoders feed in
formula = (
    "GINEConv: how atom & bond encoders feed into message passing\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "BEFORE message passing (runs once):\n"
    "  h_v  = AtomEncoder(x_v)      x_v ∈ ℝ¹⁴ → h_v ∈ ℝ¹²⁸\n"
    "  e_uv = BondEncoder(b_uv)     b_uv ∈ ℝ⁶  → e_uv ∈ ℝ¹²⁸\n"
    "\n"
    "INSIDE each GINEConv layer (runs N times):\n"
    "  For each neighbor u of atom v:\n"
    "    msg_u = ReLU( h_u + e_uv )      ← add bond info to neighbor\n"
    "  agg   = Σ_u  msg_u                ← sum all neighbor messages\n"
    "  h_v'  = MLP( (1+ε)·h_v + agg )   ← update with self + messages\n"
    "  h_v'  = BatchNorm → ReLU → Dropout → h_v + h_v'  (residual)"
)
ax2.text(0.0, -1.05, formula, fontsize=7.5, fontfamily="monospace",
         ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffff0", edgecolor="#cccc00",
                   alpha=0.95, linewidth=1.5))

ax2.set_xlim(-2.2, 2.2)
ax2.set_ylim(-2.8, 2.8)
ax2.set_aspect("equal")
ax2.axis("off")

plt.tight_layout()
plt.savefig("/home/zhedd/molgate/data/gnn_architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: gnn_architecture.png")


# ===========================================================================
# Figure 2: Forward pass — showing encoders as parallel inputs to MP
# ===========================================================================
fig2, ax = plt.subplots(figsize=(18, 7))
ax.set_title("MoleculeGNN Forward Pass: SMILES → Prediction", fontsize=15, fontweight="bold", pad=20)

# Layout coordinates — the key change is that Atom Encoder and Bond Encoder
# are vertically stacked (parallel), both feeding arrows into GINEConv.
#
#   SMILES → [featurizer] → Atom features (8×14) → [Atom Encoder] → h (8×128) ──┐
#                          → Bond features (16×6) → [Bond Encoder] → e (16×128) ─┤
#                                                                                 ▼
#                                                                         [GINEConv ×3]
#                                                                                 │
#                                                                         [Global Pool]
#                                                                                 │
#                                                                          [MLP Head]
#                                                                                 │
#                                                                              ŷ = -2.31

def draw_box(ax, x, y, w, h, label, color, desc=None, dim=None, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08", facecolor=color, edgecolor="#333333", linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    if desc:
        ax.text(x, y - h/2 - 0.15, desc, ha="center", va="top", fontsize=7, color="#555555")
    if dim:
        ax.text(x, y + h/2 + 0.1, dim, ha="center", va="bottom", fontsize=7.5,
                color="#0055AA", fontstyle="italic")

def draw_arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8, mutation_scale=15))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my + 0.1, label, fontsize=7, color="#0055AA", fontstyle="italic")

bw = 1.8  # box width
bh = 0.8  # box height

# Row 1 (top): Atom path
y_atom = 1.5
# Row 2 (bottom): Bond path
y_bond = -0.3
# Merged path
y_merge = 0.6  # vertical center for GINEConv and beyond
x_positions = [0, 2.5, 5.5, 9, 12, 14.5]

# --- SMILES input ---
draw_box(ax, x_positions[0], y_merge, bw, 1.8, "SMILES\n→\nFeaturizer", "#e8e8e8",
         desc="Parse molecule\ninto graph", fontsize=9)

# --- Atom features (raw) ---
draw_box(ax, x_positions[1], y_atom, bw, bh, "Atom Features\n(raw)", "#ddeeff",
         dim="8 atoms × 14")
draw_arrow(ax, x_positions[0] + bw/2, y_merge + 0.3, x_positions[1] - bw/2, y_atom)

# --- Bond features (raw) ---
draw_box(ax, x_positions[1], y_bond, bw, bh, "Bond Features\n(raw)", "#ddeeff",
         dim="16 edges × 6")
draw_arrow(ax, x_positions[0] + bw/2, y_merge - 0.3, x_positions[1] - bw/2, y_bond)

# --- Atom Encoder ---
draw_box(ax, 4.0, y_atom, bw, bh, "Atom Encoder\nLinear(14→128)", "#b3d9ff",
         dim="8 × 128 = h")
draw_arrow(ax, x_positions[1] + bw/2, y_atom, 4.0 - bw/2, y_atom)

# --- Bond Encoder ---
draw_box(ax, 4.0, y_bond, bw, bh, "Bond Encoder\nLinear(6→128)", "#b3d9ff",
         dim="16 × 128 = e")
draw_arrow(ax, x_positions[1] + bw/2, y_bond, 4.0 - bw/2, y_bond)

# --- GINEConv (receives both h and e) ---
gine_x = 7.0
draw_box(ax, gine_x, y_merge, 2.0, 2.0,
         "GINEConv ×3\n+ BatchNorm\n+ ReLU\n+ Residual", "#ffccb3",
         dim="8 × 128 (refined)", fontsize=9)

# Arrows from BOTH encoders into GINEConv — this is the key visual
# Atom embeddings → GINEConv (from top-right)
draw_arrow(ax, 4.0 + bw/2, y_atom, gine_x - 1.0, y_merge + 0.5, label="h (node features)")
# Bond embeddings → GINEConv (from bottom-right)
draw_arrow(ax, 4.0 + bw/2, y_bond, gine_x - 1.0, y_merge - 0.5, label="e (edge features)")

# Annotation: what happens inside
inside_text = (
    "Inside each layer:\n"
    "msg = ReLU(h_neighbor + e_bond)\n"
    "h_new = MLP((1+ε)·h_self + Σ msg)"
)
ax.text(gine_x, y_merge - 1.4, inside_text, ha="center", va="top", fontsize=7,
        fontfamily="monospace", color="#884400",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5ee", edgecolor="#cc8844", alpha=0.9))

# --- Global Mean Pool ---
pool_x = 10.0
draw_box(ax, pool_x, y_merge, bw, bh, "Global\nMean Pool", "#c2f0c2",
         dim="1 × 128", desc="Average all atom\nvectors → 1 molecule vector")
draw_arrow(ax, gine_x + 1.0, y_merge, pool_x - bw/2, y_merge)

# --- MLP Head ---
head_x = 12.5
draw_box(ax, head_x, y_merge, bw, bh, "MLP Head\n128→128→1", "#f0c2f0",
         dim="1 × 1", desc="Non-linear map\nto scalar prediction")
draw_arrow(ax, pool_x + bw/2, y_merge, head_x - bw/2, y_merge)

# --- Output ---
out_x = 14.8
draw_box(ax, out_x, y_merge, 1.2, bh, "ŷ = -2.31", "#e8e8e8")
draw_arrow(ax, head_x + bw/2, y_merge, out_x - 0.6, y_merge)

ax.set_xlim(-1.5, 16)
ax.set_ylim(-2.5, 3.0)
ax.axis("off")

plt.tight_layout()
plt.savefig("/home/zhedd/molgate/data/gnn_forward_pass.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: gnn_forward_pass.png")
