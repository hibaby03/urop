# G1-29 오른팔 전방 들어 올리기 (arm_sdk)

`g1_right_arm_raise.py`는 `unitree_sdk2py`만 사용해 `rt/arm_sdk`로 명령을 보낸다.
`xr_teleoperate`, IK, Quest, ZED, Dex3 제어 프로세스는 필요하지 않다.
현재 저장소의 G1-29 구성을 대상으로 하며 G1-23/R1/H1용 예제가 아니다.

## 실행

로봇에 연결된 PC2에서 기존 Unitree SDK/Cyclone DDS Python 환경을 사용한다.
저장소 루트에서 다음과 같이 실행한다. PC2의 로봇 연결 인터페이스가 `eth0`가
아니라면 실제 이름으로 바꾼다. IP 주소를 인자로 넣는 방식이 아니다.

```bash
git pull --ff-only

# 목표만 표시: SDK 설치와 로봇 연결 불필요, DDS 초기화/명령 발행 없음
python3 g1_right_arm_raise.py

# 실제 실행: 먼저 작은 30도 목표로 확인
python3 g1_right_arm_raise.py --interface eth0 --raise-deg 30 --execute

# 오른쪽 상완을 앞으로 약 90도, 팔꿈치는 20도 굽힘
python3 g1_right_arm_raise.py --interface eth0 --execute
```

실행 전에 G1을 고수준 균형 유지 모드에서 안정적으로 세워 둔다. 팔의 이동 공간을
확보하고 리모컨의 정지 수단을 준비한다. 녹화기, 정책 실행기, 다른 arm_sdk 예제 등
팔 명령을 보내는 프로세스는 종료한다. 이 코드는 균형 유지 모드로 자동 전환하지 않으며
motion switcher의 `ReleaseMode()`를 호출하지 않는다. 다리 제어는 로봇의 고수준 제어기가 맡는다.

## 동작 순서와 목표

1. 최대 10초간 유효한 `rt/lowstate` 수신을 기다리고 시작 자세를 저장한다.
2. 시작 자세를 유지하면서 2초 동안 arm_sdk 가중치를 0에서 1로 올린다.
3. 오른팔을 목표로 이동한다. 기본 이동 시간은 6초이며 관절 목표 속도 제한에 따라 연장한다.
4. 기본 2초간 유지한다.
5. 같은 시간에 걸쳐 시작 자세로 돌아온다.
6. 2초 동안 가중치를 1에서 0으로 낮추고 종료한다.

오른팔 목표는 아래와 같다. 모든 각도는 절대 관절 목표이며 현재 각도에 더하는 값이 아니다.
`--raise-deg`는 손끝의 정확한 높이/방향이 아닌 어깨 pitch를 지정한다. 몸통 자세와
팔꿈치 굽힘 때문에 손끝이 정확히 수평이라는 보장은 없다.

| 관절 | motor index | 기본 목표 |
|---|---:|---:|
| 오른쪽 shoulder pitch | 22 | -90도 (`--raise-deg`의 음수) |
| 오른쪽 shoulder roll | 23 | -0.1 rad (약 -5.7도) |
| 오른쪽 shoulder yaw | 24 | 0 |
| 오른쪽 elbow | 25 | +20도 (`--elbow-deg`) |
| 오른쪽 wrist roll/pitch/yaw | 26/27/28 | 0 |

G1의 shoulder pitch 축 정의에서 음수 방향으로 회전하면 팔이 앞으로 올라간다.
왼팔과 허리(12~21번)는 시작할 때 측정한 관절 위치를 계속 목표로 보낸다.
arm_sdk의 제어 가중치(29번 `q`)가 상체에 공통으로 적용되므로 함께 위치를 유지한다.
다리 관절에는 활성 제어 명령을 설정하지 않는다.

50 Hz로 `mode_pr=0`, 수신한 `mode_machine`, PD 위치 목표와 CRC를 전송한다.
어깨·팔꿈치·허리 `kp=60`, 손목 `kp=40`, 전체 `kd=1.5`이며 `dq=0`, `tau=0`이다.
중력 보상 feed-forward는 없으므로 실제 위치에 오차가 생길 수 있다.
부드러운 cubic smoothstep을 사용하며 목표 관절 속도 상한은 0.4 rad/s이다.
이 값은 명령 궤적의 제한이며 실제 모터 속도를 검증한 결과가 아니다.
프로그램이 지연되어도 누락된 시간을 따라잡기 위해 목표를 크게 건너뛰지 않는다.

```bash
# 이동을 더 느리게, 유지 시간을 짧게 지정
python3 g1_right_arm_raise.py --interface eth0 --raise-deg 45 \
  --elbow-deg 20 --move-seconds 10 --hold-seconds 1 --execute
```

## 중단과 수신 오류

팔이 전혀 움직이지 않으면 먼저 모터 명령 없이 현재 상태와 제어 서비스를 조회한다.

```bash
python3 g1_right_arm_raise.py --interface eth0 --probe
```

`mode_machine`은 LowState의 기계 구성 모드이며 고수준 동작 FSM 번호와 같은 값이 아니다.
고수준 상태는 별도의 `loco_fsm` 조회 결과에서 확인한다. `motion_switcher`와 `loco_fsm`
결과의 `code`가 0이 아니거나 오류가 나오면 해당 서비스 조회가 실패한 것이다.
조회 실패만으로 arm_sdk가 고장이라고 단정할 수는 없다. 이 진단 옵션은 팔 명령
발행기를 생성하지 않고 `CheckMode()`/`GetFsmId()`만 호출하며 제어 모드를 변경하지 않는다.

실제 실행에서는 매초 `[feedback]` 줄에 제어 가중치, 어깨의 명령 각도(`cmd`)와
측정 각도(`actual`)가 표시된다. 목표 변화가 5도보다 큰데 들어 올리기 구간이 끝날 때
실측 어깨 변화가 1도 미만이면 `NO_ARM_RESPONSE` 오류로 중단하고 제어 해제를 시도한다.
이 오류는 원인 확정이 아니라 무반응 감지이다. 실제 실행 명령, 전체 출력, 로봇의 현재
고수준 모드와 SDK/펌웨어 버전을 함께 확인해야 한다. `Done`은 전체 궤적 전송의 종료이며
모든 관절이 목표 각도에 정확하게 도달했다는 인증은 아니다.

`Ctrl+C`, 잘못된 관절 값, 로봇 모드 변경, 0.25초 이상 lowstate tick 진행 중단,
DDS 쓰기 실패가 발생하면 이동을 중단하고 가중치 0 명령을 5회 시도한 후 종료한다.
중단 시 시작 자세로 되돌아가는 추가 이동은 하지 않는다. 중복/역순 tick은
수신 제한 시간을 갱신하지 않는다. 연결이 끊기면 해제 명령 전달도 보장되지 않는다.
이 처리는 리모컨 비상 정지를 대체하지 않는다. 정상 종료 후에도 고수준 제어기가
자체 팔 자세로 움직일 수 있다. `kill -9`나 전원 중단에는 종료 처리가 실행되지 않는다.

수신 대기 중 실패하면 모터 명령을 전혀 발행하지 않는다. `No G1 lowstate`가 나오면
먼저 `python3 lowstate_probe.py --interface eth0 --domain 0 --seconds 10`으로 확인한다.
수신 성공은 arm_sdk 명령이 실제로 적용된다는 뜻은 아니다. 팔이 움직이지 않으면
고수준 모드, 다른 발행기, 해당 펌웨어의 arm_sdk 지원을 확인한다.

## 구현 근거와 검증 범위

- [Unitree 공식 G1 arm7 SDK 예제](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/high_level/g1_arm7_sdk_dds_example.py):
  토픽, 메시지, 관절 인덱스, 가중치와 CRC 방식.
- [Unitree 공식 G1-29 URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof.urdf): 관절 축과 범위.

로컬 검증은 SDK 없이 수행하는 궤적·명령 메시지·수신 중단·오류 종료 테스트이다.
실제 로봇 동작과 펌웨어 호환성은 이 개발 환경에서 검증하지 않았다.
