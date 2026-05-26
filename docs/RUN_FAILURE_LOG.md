# Run Failure Log

실행 실수와 재발 방지 절차를 따로 기록한다. 이 문서는 설계 문서가 아니라 operational mistake log다.

## 2026-05-25 residual scale 1 launch

상황:

- branch: `main`
- base commit before local edits: `6456944 Split residual RLPD agent`
- target config: `config/tool_hang_residual_bcflow_top200_mixed50_shiftm1_nstep5_scale1_seed0_2m.yaml`
- intended run: ToolHang residual RLPD, frozen top200 BCFlow base, residual scale `1.0`, GPU 6, 2M steps.

겪은 문제:

- `nohup conda run -n il python -m il.train ... &`로 background launch를 먼저 시도했다.
- PID file은 생성됐지만 실제 `il.train` 프로세스가 남지 않았다.
- log file은 비어 있었고, `nvidia-smi`에서도 GPU 6 memory 사용이 증가하지 않았다.
- `conda run` 대신 `/home/junhyeong/miniconda3/envs/il/bin/python` 직접 호출로 다시 시도했지만 같은 증상으로 바로 종료됐다.
- foreground `--build-only` 검증을 background launch 전에 먼저 끝내지 않았다.

왜 문제인가:

- PID file 생성은 학습 시작의 증거가 아니다.
- log가 비어 있고 GPU memory가 안 잡히면 run이 시작됐다고 보고하면 안 된다.
- long run은 config/load/build 검증과 짧은 foreground smoke가 끝난 뒤에만 detach해야 한다.

다음부터 지킬 절차:

1. `--build-only`를 foreground로 먼저 실행한다.
2. 100-step 이하 smoke run을 foreground 또는 짧은 detached run으로 실행하고 log/stdout을 확인한다.
3. `tail logs/...`, `pgrep -af ...`, `nvidia-smi` 세 가지가 모두 일관되게 살아있음을 확인한다.
4. PID file만 보고 "돌아간다"고 판단하지 않는다.
5. background launch 직후 로그가 비어 있으면 최소 10초 후 재확인하고, 그래도 비어 있으면 foreground로 재현한다.
6. `conda run` detached가 조용히 죽는 패턴이 있으면 env Python 직접 호출로 바꾸되, 이 역시 foreground smoke를 통과한 뒤 사용한다.

현재 상태:

- scale1 config 파일은 생성돼 있다.
- foreground `--build-only`는 통과했다.
- background train은 정상 시작 확인 전이며, 이 기록 작성 후 자동 재launch하지 않았다.
- 관련 residual 구현 변경은 아직 commit되지 않았다.
