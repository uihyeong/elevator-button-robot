"""
Isaac Sim Script Editor 실행용
알루미늄 프레임(프로파일 4개 + top plate)을 Scout Mini base_link에 고정

실행 방법:
  Isaac Sim → Window → Script Editor → 내용 붙여넣기 → Run
  (setup_mobile_manipulator.py 실행 완료 후 실행)
"""

import omni.usd
from pxr import UsdPhysics, Sdf

stage = omni.usd.get_context().get_stage()

SCOUT_PATH = "/World/scout_mini"
BASE_LINK  = SCOUT_PATH + "/base_link"

# FixedJoint로 연결할 프레임 구조물
FRAME_PARTS = [
    SCOUT_PATH + "/profile_1",
    SCOUT_PATH + "/profile_2",
    SCOUT_PATH + "/profile_3",
    SCOUT_PATH + "/profile_4",
    SCOUT_PATH + "/top_plate",
]

# base_link 존재 확인
if not stage.GetPrimAtPath(BASE_LINK).IsValid():
    print(f"[ERROR] base_link 없음: {BASE_LINK}")
    print("Scout Mini prim 구조:")
    scout = stage.GetPrimAtPath(SCOUT_PATH)
    for child in scout.GetChildren():
        print(f"  {child.GetPath()}")
    raise SystemExit

for part_path in FRAME_PARTS:
    prim = stage.GetPrimAtPath(part_path)
    if not prim.IsValid():
        print(f"[SKIP] {part_path} 없음")
        continue

    # RigidBodyAPI 적용 (physics 대상으로 등록)
    UsdPhysics.RigidBodyAPI.Apply(prim)

    # CollisionAPI — mesh 자식에 적용
    mesh = stage.GetPrimAtPath(part_path + "/mesh")
    if mesh.IsValid():
        UsdPhysics.CollisionAPI.Apply(mesh)

    # FixedJoint: Scout Mini base_link → 각 구조물
    joint_path = SCOUT_PATH + f"/joint_to_{prim.GetName()}"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BASE_LINK)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(part_path)])

    print(f"  FixedJoint: base_link → {prim.GetName()}")

print("\n✓ 완료! Play 눌러서 Scout Mini 움직이면 프레임도 같이 움직이는지 확인하세요.")
