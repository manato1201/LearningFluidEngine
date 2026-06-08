"""
mcp_setup.py  —  MCP 経由で Blender にセットアップを送るスクリプト

Claude Code (FleetView) の mcp__blender__execute_blender_code ツールに
チャンクに分けて送信するためのヘルパースクリプト。

直接実行は不要。
各チャンクのコードを順番に execute_blender_code へ渡します。
"""

# ──────────────────────────────────────────
# CHUNK 1: ライブラリ読み込み & データロード
# ──────────────────────────────────────────
CHUNK_1 = r"""
import bpy, numpy as np, mathutils
from pathlib import Path

DATA_PATH  = r"C:\Users\matuu\Desktop\GameDevelopment\FluidKit\blender\fluid_data.npz"
PARTICLE_R = 0.008
FLUID_COLOR = (0.05, 0.35, 0.9, 1.0)

d = np.load(DATA_PATH)
positions = d["positions"]
counts    = d["counts"]
bmin      = d["bmin"]
bmax      = d["bmax"]
fps       = float(d["fps"])
n_frames  = len(positions)

print(f"[FluidKit] データ読み込み完了: {n_frames}フレーム, 最大{counts.max()}粒子")
print(f"  bounds: min={bmin.round(3)}  max={bmax.round(3)}")
"""

# ──────────────────────────────────────────
# CHUNK 2: シーンクリア & 粒子メッシュ生成
# ──────────────────────────────────────────
CHUNK_2 = r"""
# シーンリセット
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 初期フレームの粒子でメッシュ生成
n0   = int(counts[0])
pts  = positions[0, :n0].tolist()
mesh = bpy.data.meshes.new("FluidMesh")
mesh.from_pydata(pts, [], [])
mesh.update()
obj  = bpy.data.objects.new("FluidParticles", mesh)
bpy.context.collection.objects.link(obj)
print(f"[FluidKit] 粒子メッシュ生成完了: {n0}頂点")
"""

# ──────────────────────────────────────────
# CHUNK 3: Geometry Nodes セットアップ
# ──────────────────────────────────────────
CHUNK_3 = r"""
obj = bpy.data.objects["FluidParticles"]
mod = obj.modifiers.new("FluidGN", "NODES")
ng  = bpy.data.node_groups.new("FluidGN", "GeometryNodeTree")
mod.node_group = ng
nodes = ng.nodes
links = ng.links

# ソケット定義
ng.interface.new_socket("Geometry", in_out="INPUT",  socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

# ノード生成
ni  = nodes.new("NodeGroupInput");    ni.location  = (-400, 0)
no  = nodes.new("NodeGroupOutput");   no.location  = ( 500, 0)
m2p = nodes.new("GeometryNodeMeshToPoints"); m2p.location = (-200, 0)
sph = nodes.new("GeometryNodeMeshUVSphere"); sph.location = (-200,-160)
ins = nodes.new("GeometryNodeInstanceOnPoints"); ins.location = (100, 0)
rel = nodes.new("GeometryNodeRealizeInstances");  rel.location = (300, 0)

sph.inputs["Radius"].default_value   = PARTICLE_R
sph.inputs["Segments"].default_value = 8
sph.inputs["Rings"].default_value    = 6

links.new(ni.outputs[0],  m2p.inputs["Mesh"])
links.new(m2p.outputs[0], ins.inputs["Points"])
links.new(sph.outputs[0], ins.inputs["Instance"])
links.new(ins.outputs[0], rel.inputs["Geometry"])
links.new(rel.outputs[0], no.inputs[0])
print("[FluidKit] Geometry Nodes セットアップ完了")
"""

# ──────────────────────────────────────────
# CHUNK 4: マテリアル
# ──────────────────────────────────────────
CHUNK_4 = r"""
mat = bpy.data.materials.new("FluidMat")
mat.use_nodes = True
mat.node_tree.nodes.clear()
out  = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
out.location = (300,0); bsdf.location = (0,0)
mat.node_tree.links.new(bsdf.outputs[0], out.inputs[0])

bsdf.inputs["Base Color"].default_value          = (0.05, 0.35, 0.9, 1.0)
bsdf.inputs["Roughness"].default_value           = 0.05
bsdf.inputs["IOR"].default_value                 = 1.33
bsdf.inputs["Transmission Weight"].default_value = 0.85
bsdf.inputs["Subsurface Weight"].default_value   = 0.12
bsdf.inputs["Subsurface Radius"].default_value   = (0.4, 0.8, 1.0)

bpy.data.objects["FluidParticles"].data.materials.append(mat)
print("[FluidKit] マテリアル適用完了")
"""

# ──────────────────────────────────────────
# CHUNK 5: カメラ・ライト・床
# ──────────────────────────────────────────
CHUNK_5 = r"""
center = (bmin + bmax) / 2
span   = float(np.linalg.norm(bmax - bmin))

# カメラ
bpy.ops.object.camera_add(location=(
    center[0] + span*0.9,
    center[1] - span*1.6,
    center[2] + span*0.7,
))
cam = bpy.context.object; cam.name = "FluidCam"
bpy.context.scene.camera = cam
direction = mathutils.Vector(center.tolist()) - cam.location
cam.rotation_euler = direction.to_track_quat("-Z","Y").to_euler()
cam.data.lens = 50

# キーライト
bpy.ops.object.light_add(type="AREA",
    location=(center[0]+span, center[1]+span*0.3, center[2]+span*1.5))
kl = bpy.context.object; kl.name = "KeyLight"
kl.data.energy = 800; kl.data.size = span*1.5
kl.data.color  = (0.9,0.95,1.0)

# フィルライト
bpy.ops.object.light_add(type="AREA",
    location=(center[0]-span*0.8, center[1], center[2]+span*0.5))
fl = bpy.context.object; fl.name = "FillLight"
fl.data.energy = 200; fl.data.size = span
fl.data.color  = (0.6,0.7,1.0)

# 床
bpy.ops.mesh.primitive_plane_add(size=span*3,
    location=(center[0], center[1], float(bmin[2])-0.01))
floor = bpy.context.object; floor.name = "Floor"
fm = bpy.data.materials.new("FloorMat"); fm.use_nodes = True
fb = fm.node_tree.nodes.get("Principled BSDF")
if fb:
    fb.inputs["Base Color"].default_value = (0.85,0.87,0.9,1.0)
    fb.inputs["Roughness"].default_value  = 0.3
floor.data.materials.append(fm)

# 世界背景（薄いグラデーション）
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg: bg.inputs["Color"].default_value = (0.06, 0.07, 0.12, 1.0)

print("[FluidKit] カメラ・ライト・床 配置完了")
"""

# ──────────────────────────────────────────
# CHUNK 6: フレームハンドラ & レンダー設定
# ──────────────────────────────────────────
CHUNK_6 = r"""
# グローバル変数にデータ保持（ハンドラから参照）
import bpy

_fd_pos    = positions
_fd_counts = counts

def _fluidkit_frame_update(scene, depsgraph=None):
    fi  = max(0, min(scene.frame_current - 1, len(_fd_pos) - 1))
    n   = int(_fd_counts[fi])
    pts = _fd_pos[fi, :n]
    obj = bpy.data.objects.get("FluidParticles")
    if obj is None: return
    mesh = obj.data
    if len(mesh.vertices) == n:
        mesh.vertices.foreach_set("co", pts.flatten())
    else:
        mesh.clear_geometry()
        mesh.from_pydata(pts.tolist(), [], [])
    mesh.update()

# 既存ハンドラを削除して再登録
fcp = bpy.app.handlers.frame_change_pre
for h in list(fcp):
    if getattr(h, "__name__", "") == "_fluidkit_frame_update":
        fcp.remove(h)
fcp.append(_fluidkit_frame_update)

# レンダー設定
import os
scene = bpy.context.scene
scene.render.engine         = "CYCLES"
scene.cycles.samples        = 64
scene.cycles.use_denoising  = True
scene.render.fps            = int(fps)
scene.frame_start           = 1
scene.frame_end             = n_frames
scene.render.resolution_x   = 1280
scene.render.resolution_y   = 720
out_dir = r"C:\Users\matuu\Desktop\GameDevelopment\FluidKit\blender\render"
os.makedirs(out_dir, exist_ok=True)
scene.render.filepath       = out_dir + r"\frame_"
scene.render.image_settings.file_format = "PNG"

# GPU 設定
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    for dev in prefs.devices:
        dev.use = True
    scene.cycles.device = "GPU"
    print("[FluidKit] GPU レンダリング: CUDA 有効")
except Exception as e:
    print(f"[FluidKit] GPU 設定スキップ: {e}")

# フレーム 1 を適用
bpy.context.scene.frame_set(1)

print("[FluidKit] ══ セットアップ完了 ══")
print("  タイムラインを再生 → 流体アニメーション確認")
print("  Ctrl+F12 → アニメーションレンダリング開始")
"""
