"""
FluidKit — Blender セットアップスクリプト
=========================================
このスクリプトを Blender の Scripting タブから実行するか、
MCP 経由で execute_blender_code に渡してください。

機能:
  - fluid_data.npz を読み込み
  - Geometry Nodes で粒子を球としてインスタンス表示
  - frame_change_pre ハンドラでフレームごとに座標を更新
  - 流体マテリアル（サブサーフェス散乱）をセットアップ
  - HDRI ライティング + カメラを配置

実行前に fluid_data.npz のパスを DATA_PATH に設定してください。
"""

import bpy
import numpy as np
from pathlib import Path
import mathutils

# ══════════════════════════════════════════════
#  設定
# ══════════════════════════════════════════════

DATA_PATH   = r"C:\Users\matuu\Desktop\GameDevelopment\FluidKit\blender\fluid_data.npz"
PARTICLE_R  = 0.008      # 球のスケール（データ座標系に合わせて調整）
FLUID_COLOR = (0.05, 0.35, 0.9, 1.0)   # RGBA — 水色
RENDER_ENGINE = "CYCLES"   # "CYCLES" or "BLENDER_EEVEE"
RENDER_SAMPLES = 64


# ══════════════════════════════════════════════
#  ① データ読み込み
# ══════════════════════════════════════════════

def load_data(path: str):
    d = np.load(path)
    positions = d["positions"]   # (F, N, 3) float32
    counts    = d["counts"]      # (F,)      int32
    bmin      = d["bmin"]        # (3,)
    bmax      = d["bmax"]        # (3,)
    fps       = float(d["fps"])
    return positions, counts, bmin, bmax, fps


# ══════════════════════════════════════════════
#  ② シーンをリセット
# ══════════════════════════════════════════════

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


# ══════════════════════════════════════════════
#  ③ 流体メッシュオブジェクト生成
# ══════════════════════════════════════════════

def create_particle_mesh(name: str, positions_frame0: np.ndarray, n: int):
    """頂点 = 粒子位置のメッシュオブジェクトを生成。"""
    pts = positions_frame0[:n].tolist()
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(pts, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# ══════════════════════════════════════════════
#  ④ Geometry Nodes — 球インスタンス
# ══════════════════════════════════════════════

def add_geometry_nodes(obj, radius: float):
    """各頂点に UV 球をインスタンス化する Geometry Nodes を追加。"""
    mod = obj.modifiers.new("FluidParticles", "NODES")
    node_group = bpy.data.node_groups.new("FluidGN", "GeometryNodeTree")
    mod.node_group = node_group
    nodes = node_group.nodes
    links = node_group.links

    # ノードを追加
    n_input   = nodes.new("NodeGroupInput")
    n_output  = nodes.new("NodeGroupOutput")
    n_points  = nodes.new("GeometryNodeMeshToPoints")
    n_sphere  = nodes.new("GeometryNodeMeshUVSphere")
    n_inst    = nodes.new("GeometryNodeInstanceOnPoints")
    n_realize = nodes.new("GeometryNodeRealizeInstances")

    # I/O ソケット
    node_group.interface.new_socket("Geometry", in_out="INPUT",  socket_type="NodeSocketGeometry")
    node_group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    # 球サイズ設定
    n_sphere.inputs["Radius"].default_value    = radius
    n_sphere.inputs["Segments"].default_value  = 8
    n_sphere.inputs["Rings"].default_value     = 6

    # レイアウト
    n_input.location   = (-400, 0)
    n_points.location  = (-200, 0)
    n_sphere.location  = (-200, -160)
    n_inst.location    = (0, 0)
    n_realize.location = (200, 0)
    n_output.location  = (400, 0)

    # 接続
    links.new(n_input.outputs[0],    n_points.inputs["Mesh"])
    links.new(n_points.outputs[0],   n_inst.inputs["Points"])
    links.new(n_sphere.outputs[0],   n_inst.inputs["Instance"])
    links.new(n_inst.outputs[0],     n_realize.inputs["Geometry"])
    links.new(n_realize.outputs[0],  n_output.inputs[0])

    return mod


# ══════════════════════════════════════════════
#  ⑤ 流体マテリアル
# ══════════════════════════════════════════════

def create_fluid_material(name: str, color: tuple):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out   = nodes.new("ShaderNodeOutputMaterial")
    bsdf  = nodes.new("ShaderNodeBsdfPrincipled")

    # 水らしいパラメータ
    bsdf.inputs["Base Color"].default_value         = color
    bsdf.inputs["Roughness"].default_value          = 0.05
    bsdf.inputs["IOR"].default_value                = 1.33    # 水の屈折率
    bsdf.inputs["Transmission Weight"].default_value = 0.85   # 透過
    bsdf.inputs["Subsurface Weight"].default_value   = 0.15
    bsdf.inputs["Subsurface Radius"].default_value   = (0.4, 0.8, 1.0)

    out.location  = (300, 0)
    bsdf.location = (0, 0)
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat


# ══════════════════════════════════════════════
#  ⑥ フレームハンドラ（アニメーション更新）
# ══════════════════════════════════════════════

_FLUID_DATA  = {}   # グローバルにデータを保持

def _on_frame_change(scene, depsgraph=None):
    """フレームが変わるたびに頂点座標を更新。"""
    data = _FLUID_DATA
    if "positions" not in data:
        return

    frame = scene.frame_current - 1          # 0-indexed
    frame = max(0, min(frame, len(data["positions"]) - 1))
    n = int(data["counts"][frame])
    pts = data["positions"][frame, :n]

    obj = bpy.data.objects.get("FluidParticles")
    if obj is None or obj.type != "MESH":
        return

    mesh = obj.data
    if len(mesh.vertices) == n:
        # 頂点数が同じ → 座標だけ更新（高速）
        flat = pts.flatten()
        mesh.vertices.foreach_set("co", flat)
    else:
        # 頂点数が変わった → メッシュを再構築
        new_pts = pts.tolist()
        mesh.clear_geometry()
        mesh.from_pydata(new_pts, [], [])
    mesh.update()


def register_handler():
    """frame_change_pre ハンドラを登録（二重登録防止）。"""
    handlers = bpy.app.handlers.frame_change_pre
    for h in handlers:
        if h.__name__ == "_on_frame_change":
            handlers.remove(h)
    handlers.append(_on_frame_change)
    print("[FluidKit] フレームハンドラ登録完了")


# ══════════════════════════════════════════════
#  ⑦ カメラ & ライティング
# ══════════════════════════════════════════════

def setup_camera_and_lights(bmin, bmax):
    center = (bmin + bmax) / 2
    span   = np.linalg.norm(bmax - bmin)

    # カメラ
    bpy.ops.object.camera_add(location=(
        center[0] + span * 0.8,
        center[1] - span * 1.5,
        center[2] + span * 0.6,
    ))
    cam = bpy.context.object
    cam.name = "FluidCamera"
    bpy.context.scene.camera = cam

    # カメラをシーン中心に向ける
    direction = mathutils.Vector(center.tolist()) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 50

    # キーライト
    bpy.ops.object.light_add(type="AREA", location=(
        center[0] + span,
        center[1] + span * 0.3,
        center[2] + span * 1.5,
    ))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 800
    key.data.size   = span * 1.5
    key.data.color  = (0.9, 0.95, 1.0)

    # フィルライト（反対側）
    bpy.ops.object.light_add(type="AREA", location=(
        center[0] - span * 0.8,
        center[1],
        center[2] + span * 0.5,
    ))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 200
    fill.data.size   = span
    fill.data.color  = (0.6, 0.7, 1.0)

    # 床
    bpy.ops.mesh.primitive_plane_add(size=span * 3, location=(
        center[0], center[1], bmin[2] - 0.01
    ))
    floor = bpy.context.object
    floor.name = "Floor"
    floor_mat = bpy.data.materials.new("FloorMat")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.87, 0.9, 1.0)
        bsdf.inputs["Roughness"].default_value  = 0.3
    floor.data.materials.append(floor_mat)


# ══════════════════════════════════════════════
#  ⑧ レンダー設定
# ══════════════════════════════════════════════

def setup_render(fps: float, n_frames: int, engine: str, samples: int):
    scene = bpy.context.scene
    scene.render.engine          = engine
    scene.render.fps             = int(fps)
    scene.frame_start            = 1
    scene.frame_end              = n_frames
    scene.render.resolution_x    = 1280
    scene.render.resolution_y    = 720
    scene.render.filepath        = r"C:\Users\matuu\Desktop\GameDevelopment\FluidKit\blender\render\frame_"
    scene.render.image_settings.file_format = "PNG"

    if engine == "CYCLES":
        scene.cycles.samples         = samples
        scene.cycles.use_denoising  = True
        # GPU レンダリング有効化
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            cprefs = prefs.preferences
            cprefs.compute_device_type = "CUDA"
            for dev in cprefs.devices:
                dev.use = True
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = samples


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════

def main():
    print("=" * 50)
    print("[FluidKit] Blender セットアップ開始")
    print("=" * 50)

    # データ読み込み
    print(f"  データ読み込み: {DATA_PATH}")
    positions, counts, bmin, bmax, fps = load_data(DATA_PATH)
    n_frames = len(positions)
    print(f"  フレーム数: {n_frames}, 最大粒子数: {counts.max()}, FPS: {fps}")

    # グローバルに保持（ハンドラから参照）
    _FLUID_DATA["positions"] = positions
    _FLUID_DATA["counts"]    = counts

    # シーンリセット
    print("  シーンをクリア...")
    clear_scene()

    # 粒子メッシュ生成（初期フレーム）
    print("  粒子メッシュ生成...")
    n0 = int(counts[0])
    obj = create_particle_mesh("FluidParticles", positions[0], n0)

    # Geometry Nodes セットアップ
    print("  Geometry Nodes セットアップ...")
    add_geometry_nodes(obj, PARTICLE_R)

    # マテリアル適用
    print("  マテリアル作成...")
    mat = create_fluid_material("FluidMat", FLUID_COLOR)
    obj.data.materials.append(mat)

    # カメラ & ライティング
    print("  カメラ & ライティング配置...")
    setup_camera_and_lights(bmin, bmax)

    # レンダー設定
    print("  レンダー設定...")
    setup_render(fps, n_frames, RENDER_ENGINE, RENDER_SAMPLES)

    # フレームハンドラ登録
    register_handler()

    # フレーム 1 に戻して初期状態を反映
    bpy.context.scene.frame_set(1)

    print("=" * 50)
    print("[FluidKit] セットアップ完了!")
    print(f"  → タイムラインを再生してシミュレーションを確認してください")
    print(f"  → Ctrl+F12 でアニメーションレンダリング開始")
    print("=" * 50)


main()
