"""
Isaac Sim Script Editor 실행용
Scout Mini 위에 알루미늄 프로파일 4개 + OpenMANIPULATOR-X 올리기

실행 방법:
  Isaac Sim → Window → Script Editor → 내용 붙여넣기 → Run
  (이전 prim 있으면 Stage에서 profile_*, top_plate, open_manipulator_x 삭제 후 실행)
"""

import omni.usd
from pxr import UsdGeom, UsdShade, Gf, Sdf, Usd

stage = omni.usd.get_context().get_stage()

# ───────────────────────────────────────────────────────────────────
# 설정값
# ───────────────────────────────────────────────────────────────────
SCOUT_PATH = "/World/scout_mini"

ARM_USD = (
    "/home/sejong/colcon_ws/src/open_manipulator/"
    "open_manipulator_x_description/urdf/open_manipulator_x/"
    "open_manipulator_x.usd"
)

PROFILE_W = 0.04   # 40mm
PROFILE_D = 0.04   # 40mm
PROFILE_H = 0.65   # 650mm

# 프로파일 4개 최종 위치 (scout_mini 로컬 좌표)
# xform 위치 + 이전 mesh 로컬 오프셋 합산값
PILLAR_XY = [
    ( 0.15579,  0.11845),   # 프로파일 1
    ( 0.01910,  0.00135),   # 프로파일 2
    (-0.15387,  0.11381),   # 프로파일 3
    (-0.15291, -0.11394),   # 프로파일 4
]

TOP_PLATE_H = 0.005   # 상단 연결 플레이트 두께 5mm

# 로봇팔 위치 (scout_mini 로컬 좌표, z는 top plate 상단으로 자동 계산)
ARM_X = 0.12757
ARM_Y = 0.0

MANUAL_TOP_Z = 0.32   # BBox 실패 시 수동 지정

# ───────────────────────────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────────────────────────
def make_aluminum_mat(stage, path):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/sh")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.25)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.78, 0.78, 0.82)
    )
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

def add_box(stage, prim_path, xyz, size_xyz, mat):
    xf = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.Xformable(xf).AddTranslateOp().Set(Gf.Vec3d(*xyz))
    cube = UsdGeom.Cube.Define(stage, prim_path + "/mesh")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*size_xyz))
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

# ───────────────────────────────────────────────────────────────────
# Scout Mini 상단 Z (로컬 좌표)
# ───────────────────────────────────────────────────────────────────
scout_prim = stage.GetPrimAtPath(SCOUT_PATH)
if not scout_prim.IsValid():
    print(f"[ERROR] '{SCOUT_PATH}' 를 찾을 수 없습니다.")
    for p in stage.GetPseudoRoot().GetChildren():
        print(f"  {p.GetPath()}")
    raise SystemExit

try:
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world_bbox  = bbox_cache.ComputeWorldBound(scout_prim)
    world_top_z = world_bbox.GetBox().GetMax()[2]
    scout_mat   = UsdGeom.Xformable(scout_prim).ComputeLocalToWorldTransform(
                      Usd.TimeCode.Default())
    scout_wz    = scout_mat.ExtractTranslation()[2]
    local_top_z = world_top_z - scout_wz
    if abs(world_top_z) < 0.001:
        raise ValueError("BBox Z ≈ 0")
    print(f"Scout Mini 상단 (world): {world_top_z:.4f} m  →  로컬: {local_top_z:.4f} m")
except Exception as e:
    local_top_z = MANUAL_TOP_Z
    print(f"[경고] BBox 실패 ({e}), 수동값 사용: {local_top_z} m")

pillar_center_z = local_top_z + PROFILE_H / 2
top_plate_z     = local_top_z + PROFILE_H + TOP_PLATE_H / 2
arm_z           = local_top_z + PROFILE_H + TOP_PLATE_H

# ───────────────────────────────────────────────────────────────────
# 재질
# ───────────────────────────────────────────────────────────────────
al_mat = make_aluminum_mat(stage, SCOUT_PATH + "/materials/aluminum")

# ───────────────────────────────────────────────────────────────────
# 알루미늄 프로파일 4개
# ───────────────────────────────────────────────────────────────────
for i, (px, py) in enumerate(PILLAR_XY):
    add_box(
        stage,
        prim_path=SCOUT_PATH + f"/profile_{i+1}",
        xyz=(px, py, pillar_center_z),
        size_xyz=(PROFILE_W, PROFILE_D, PROFILE_H),
        mat=al_mat,
    )
    print(f"  프로파일 {i+1}: ({px:+.5f}, {py:+.5f}, {pillar_center_z:.4f})")

# ───────────────────────────────────────────────────────────────────
# 상단 플레이트 (4기둥 범위를 감싸도록 자동 크기 계산)
# ───────────────────────────────────────────────────────────────────
xs = [p[0] for p in PILLAR_XY]
ys = [p[1] for p in PILLAR_XY]
plate_cx = (max(xs) + min(xs)) / 2
plate_cy = (max(ys) + min(ys)) / 2
plate_w  = max(xs) - min(xs) + PROFILE_W
plate_d  = max(ys) - min(ys) + PROFILE_D

add_box(
    stage,
    prim_path=SCOUT_PATH + "/top_plate",
    xyz=(plate_cx, plate_cy, top_plate_z),
    size_xyz=(plate_w, plate_d, TOP_PLATE_H),
    mat=al_mat,
)
print(f"  상단 플레이트: {plate_w*1000:.0f}x{plate_d*1000:.0f}x{TOP_PLATE_H*1000:.0f}mm  "
      f"중심=({plate_cx:.4f}, {plate_cy:.4f}, {top_plate_z:.4f})")

# ───────────────────────────────────────────────────────────────────
# OpenMANIPULATOR-X (top plate 위)
# ───────────────────────────────────────────────────────────────────
arm_prim = stage.DefinePrim(SCOUT_PATH + "/open_manipulator_x", "Xform")
arm_prim.GetReferences().AddReference(ARM_USD)
arm_xf = UsdGeom.Xformable(arm_prim)
arm_xf.ClearXformOpOrder()
arm_xf.AddTranslateOp().Set(Gf.Vec3d(ARM_X, ARM_Y, arm_z))

print(f"  로봇팔: ({ARM_X}, {ARM_Y}, {arm_z:.4f})")
print("\n✓ 완료!")
print(f"  프로파일 높이: {PROFILE_H*1000:.0f}mm  /  로봇팔 base Z: {arm_z:.4f} m")
