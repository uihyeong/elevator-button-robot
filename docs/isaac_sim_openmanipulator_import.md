# Isaac Sim에 OpenMANIPULATOR-X 불러오기

Isaac Sim에서 OpenMANIPULATOR-X(URDF)를 빈 씬에 임포트하는 전체 절차.
"씬 준비 → URDF 임포트 → 임포트 후 확인" 순서로 진행한다.

> 환경 기준: Isaac Sim 5.1.0 / Ubuntu 22.04
> URDF 경로: `~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/`

---

## 0. 왜 씬부터 만드는가

URDF Importer는 **로봇 아티큘레이션만** 만든다.
바닥(Ground)·중력(PhysicsScene)·조명은 만들어 주지 않으므로,
아무 설정 없이 임포트하면 로봇이 허공에 뜬 채로 물리 시뮬레이션이 돌지 않거나,
Play를 눌러도 관절이 힘을 못 받아 그냥 쓰러진다.

그래서 **임포트 전에 씬(무대)을 먼저 세팅**하는 것이 핵심이다.

```
[1] 빈 스테이지 → [2] PhysicsScene(중력) → [3] Ground Plane(바닥)
      → [4] 조명 → [5] URDF Import → [6] Play로 확인
```

---

## 1. 새 스테이지 생성

- `File → New Stage` (또는 `Ctrl+N`)
- 단위 확인: `Stage` 창에서 미터(m) 단위인지 확인
  - Isaac Sim 기본은 **meters per unit = 1.0** (미터). URDF도 미터 기준이라 그대로 맞다.

---

## 2. Physics Scene 추가 (중력 설정)

물리 시뮬레이션과 중력을 담당하는 prim.

- `Create → Physics → Physics Scene`
- 생성된 `/World/physicsScene` 선택 후 속성(Property) 확인:
  - **Gravity Direction**: `(0, 0, -1)` — Z축 아래 방향
  - **Gravity Magnitude**: `9.81` (m/s²)
- 물리 정확도 관련(선택):
  - **Solver Type**: `TGS` (기본, 관절 안정적)
  - **Time Steps Per Second**: `60` 또는 `120` (관절 진동 심하면 올린다)

> ⚠️ Physics Scene이 없으면 Play를 눌러도 중력·충돌이 전혀 적용되지 않는다.

---

## 3. Ground Plane 추가 (바닥)

로봇이 떨어지지 않도록 받쳐주는 충돌 바닥.

- `Create → Physics → Ground Plane`
- 생성 위치: `/World/GroundPlane`, z = 0
- 자동으로 **Collider(충돌)** 가 포함되어 있어 별도 CollisionAPI 설정 불필요
- (선택) 마찰이 필요하면 Ground Plane에 **Physics Material** 적용
  - `Create → Physics → Physics Material` → static/dynamic friction 지정 후 바닥에 바인딩

> 로봇 base를 고정(Fixed Base)해서 쓸 거면 바닥이 없어도 되지만,
> 시각적 기준·물체 낙하 테스트를 위해 두는 것을 권장.

---

## 4. 조명(Light) 추가

임포트 자체엔 필수는 아니지만, 없으면 화면이 까맣게 보인다.

- `Create → Light → Dome Light` (전체 균일 조명, 가장 편함)
- 또는 `Distant Light` (태양광 느낌, 그림자 표현)

---

## 5. URDF Import (핵심)

### 5-1. Importer 열기

- 상단 메뉴 `File → Import`
  - 또는 `Isaac Utils → Workflows → URDF Importer` (버전에 따라 위치 다름)
- URDF 파일 선택:
  ```
  ~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/open_manipulator_x.urdf
  ```
  (파일명은 레포 버전에 따라 다를 수 있으니 실제 `.urdf` 확인)

### 5-2. Import 옵션 (중요)

| 옵션 | 권장값 | 설명 |
|------|--------|------|
| **Fixed Base Link** | ✅ ON | 로봇팔 베이스를 바닥에 고정. 팔은 움직이면 안 되므로 켠다. (Scout Mini 같은 이동체는 OFF) |
| **Import Inertia Tensor** | ✅ ON | URDF의 관성값 사용 |
| **Stiffness / Damping** | 기본값 or 조정 | Position drive면 stiffness↑, Velocity drive면 stiffness=0 / damping↑ |
| **Joint Drive Type** | Position | 관절을 각도로 제어할 경우 Position drive |
| **Self Collision** | ❌ OFF | 링크끼리 충돌로 관절이 튀는 것 방지 (필요 시에만 ON) |
| **Merge Fixed Joints** | 선택 | 고정 조인트 병합. 링크 수 줄고 성능↑, 단 개별 링크 접근은 못 함 |
| **Create Physics Scene** | ❌ OFF | 2번에서 이미 만들었으면 중복 생성 방지 (없으면 ON) |
| **Density / Default Prim** | 기본 | 그대로 |

> 로봇팔(고정 베이스) vs 이동 로봇(Scout Mini) 옵션 차이
> - **OpenMANIPULATOR-X**: `Fixed Base = ON`, Position drive
> - **Scout Mini (바퀴)**: `Fixed Base = OFF (Moveable)`, **Velocity drive** (프로젝트 트러블슈팅 참고)

### 5-3. Import 실행

- `Import` 버튼 클릭
- `/World` 아래에 `open_manipulator_x` 아티큘레이션이 생성됨
- Stage 트리에서 `base_link → link1 ~ link5 → gripper` 구조 확인

---

## 6. 임포트 후 확인

### 6-1. 위치 정렬

- 로봇 base가 바닥(z=0)에 닿도록 Transform 확인
- 필요 시 `/World/open_manipulator_x` Translate z 조정

### 6-2. Articulation Root 확인

- 로봇 최상위 prim에 **ArticulationRootAPI**가 있는지 확인
- ⚠️ ArticulationRootAPI가 **2개 이상**이면 오류 → 하나만 남긴다
  - (프로젝트 사례: mobile_manipulator에서 팔 root_joint의 ArticulationRootAPI 제거 필요)

### 6-3. Play로 물리 확인

- 상단 **Play(▶️)** 클릭
- 팔이 중력에 쓰러지지 않고 **관절이 0 자세를 유지**하면 정상
  - 쓰러지면: Fixed Base가 꺼졌거나, joint drive stiffness가 0이거나, PhysicsScene이 없는 경우
  - 관절이 진동/발산하면: Time Steps Per Second 올리기 / stiffness·damping 재조정

### 6-4. USD로 저장

- `File → Save As` → `.usd`로 저장
  ```
  ~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/open_manipulator_x/open_manipulator_x.usd
  ```
- 다음부터는 URDF 재임포트 없이 이 USD를 바로 불러 쓴다.

---

## 7. 자주 겪는 문제 (트러블슈팅)

| 증상 | 원인 | 해결 |
|------|------|------|
| Play 눌러도 팔이 쓰러짐 | Fixed Base OFF / PhysicsScene 없음 | Fixed Base ON, PhysicsScene 추가 |
| 관절이 힘없이 흐물거림 | joint drive stiffness=0 | Position drive stiffness 값 부여 |
| 화면이 까맣다 | 조명 없음 | Dome Light 추가 |
| 로봇이 바닥을 뚫고 떨어짐 | Ground Plane 없음 / Collider 없음 | Ground Plane 추가 |
| `Found multiple articulations` | ArticulationRootAPI 2개 | 상위 하나만 남기고 제거 |
| 관절이 튀거나 발산 | Self Collision ON / dt 큼 | Self Collision OFF, Steps/sec 올림 |
| 메시가 안 보임 | 메시 경로 문제 | URDF의 mesh 경로/패키지 확인 |

---

## 참고

- 임포트에 사용한 URDF + 메시 원본은 [`urdf/`](urdf/) 폴더에 함께 올려두었다.
  (메시 경로를 상대경로로 바꿔 폴더만으로 바로 임포트 가능)
- 이동 로봇(Scout Mini) 임포트는 **Velocity drive + Moveable Base**로 옵션이 다르다.
  자세한 내용은 `CLAUDE.md`의 "Scout Mini USD 설정" 및 RL 트러블슈팅 참고.
- Isaac Lab(강화학습)에서 쓸 USD는 ArticulationRootAPI가 **정확히 1개**여야 한다.
