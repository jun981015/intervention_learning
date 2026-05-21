# 작업 진행 방식

이 문서는 전체 workflow의 큰 단위만 유지한다. 세부 설계와 결정 이유는 작업 단위별
문서로 분리한다.

현재 구현 상태와 검증 결과의 빠른 스냅샷은
[STATUS_2026-05-18.md](STATUS_2026-05-18.md)를 본다.

## 기준 원칙

- `qc`와 `qc_base`는 참조용이다. 이 repo는 intervention learning 전용 독립 프로젝트다.
- v0는 Robomimic `square-mh-low_dim`, RLPD/SAC learner, RLPD/SAC expert checkpoint,
  random gate, `horizon_length=1`을 기준으로 한다.
- FQL, QC-FQL, BT model, OGBench/cube, human UI, action chunking은 v0 범위 밖이다.
- learner action, expert action, executed action, gate metadata는 replay에서 분리해서 저장한다.
- weight, replay, video, log, wandb 산출물은 git에 넣지 않는다.

자세한 범위와 제외 항목은 [PROJECT_SCOPE.md](PROJECT_SCOPE.md)를 본다.

## 초기 파이프라인

큰 흐름은 learner/expert action을 둘 다 뽑고, gate가 실행 action을 고른 뒤, replay를
online/demo/intervention 역할별 buffer에 저장하는 것이다.

step logic과 buffer routing 세부 내용은 [PIPELINE.md](PIPELINE.md)를 본다.

DAgger baseline은 gate/intervention과 다른 경로다. learner action으로 env를 진행하고
expert action은 relabel target으로만 저장한다. 세부 내용은
[DAGGER_BASELINE.md](DAGGER_BASELINE.md)를 본다.

## Buffer 역할

세 buffer의 역할은 다음 문서에서 관리한다.

- `online_buffer`: [PIPELINE.md#buffer-역할](PIPELINE.md#buffer-역할)
- `demo_buffer`: [PIPELINE.md#buffer-역할](PIPELINE.md#buffer-역할)
- `intervention_buffer`: [PIPELINE.md#buffer-역할](PIPELINE.md#buffer-역할)

## N-step Backup

QC/RLPD baseline에 맞춰 `sample_sequence(batch_size, n, gamma)` API를 유지한다.

target 수식과 episode boundary 정책은
[REPLAY_AND_UPDATES.md#n-step-backup](REPLAY_AND_UPDATES.md#n-step-backup)과
[REPLAY_AND_UPDATES.md#episode-boundary-처리](REPLAY_AND_UPDATES.md#episode-boundary-처리)를 본다.

## Update-to-data Ratio

UTD는 config로 열어둔다. reshape 방식은
[REPLAY_AND_UPDATES.md#update-to-data-ratio](REPLAY_AND_UPDATES.md#update-to-data-ratio)를 본다.

여러 buffer를 섞어서 batch를 만드는 방식은
[REPLAY_AND_UPDATES.md#mixed-buffer-sampling](REPLAY_AND_UPDATES.md#mixed-buffer-sampling)을 본다.

## TD Target Q Aggregation

기본값은 `target_q_agg="min"`이다. 세부 정책은
[REPLAY_AND_UPDATES.md#td-target-q-aggregation](REPLAY_AND_UPDATES.md#td-target-q-aggregation)을 본다.

## Frame Stack

v0 기본값은 `frame_stack=1`이다. 세부 설명은
[REPLAY_AND_UPDATES.md#frame-stack](REPLAY_AND_UPDATES.md#frame-stack)을 본다.

## Transition Schema

필드 목록과 `terminals/masks` 의미는
[REPLAY_AND_UPDATES.md#transition-schema](REPLAY_AND_UPDATES.md#transition-schema)와
[REPLAY_AND_UPDATES.md#terminal과-mask](REPLAY_AND_UPDATES.md#terminal과-mask)를 본다.

## Network 기본값

공용 MLP 옵션, LayerNorm 위치, BC Flow의 flow-specific 설정은 [NETWORKS.md](NETWORKS.md)를 본다.

## 구현 순서

완료된 scaffold, 현재 smoke 결과, 다음 구현 순서는
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)를 본다.

## Git 운영

- 이 repo는 `/home/junhyeong/repos/intervention_learning`에서 독립 git으로 관리한다.
- remote는 아직 연결하지 않는다.
- GitHub SSH 설정이 끝나면 그때 remote를 추가하고 push한다.
- weight, replay, video, log, wandb 산출물은 절대 커밋하지 않는다.
- 큰 변경은 작은 커밋으로 쪼갠다.

## 참조 위치

- 현재 실험 코드: `/home/junhyeong/repos/qc`
- 원본 QC clean reference: `/home/junhyeong/repos/qc_base`
