# 구현 계획과 검증 상태


## 현재 상태 업데이트 — 2026-05-27

이 문서는 초기 구현 계획의 기록을 포함한다. 현재 작업 우선순위와 최신 TODO는
[IMPLEMENTATION_TODO.md](IMPLEMENTATION_TODO.md)를 기준으로 본다. 실제 env smoke 결과는
[REAL_ENV_SMOKE_TESTS.md](REAL_ENV_SMOKE_TESTS.md)를 보고, 2026-05-26 코드 리뷰 후속은
[CODE_REVIEW_2026-05-26.md](CODE_REVIEW_2026-05-26.md)의 현재 해결 상태 표를 본다.

초기 계획 중 Robomimic env construction, recipe-driven `il.train` entrypoint, DAgger relabel real-env
smoke, expert-Q gap real-env smoke, residual RLPD/TD3 path는 이미 구현 또는 smoke 검증이 진행됐다.
남은 작업은 replay round-trip, dataset adapter/canonicalization, gate abstraction 정리, update/checkpoint
config 반영, action chunk queue, image policy 학습 쪽이다.

## 완료된 Scaffold

- Project scaffold와 replay schema를 만들었다.
- Random gate와 policy interface를 만들었다.
- N-step `ReplayBuffer.sample_sequence()` smoke test를 만들었다.
- `qc_base` 기준 최소 RLPD/SAC agent adapter를 가져왔다.
- BC Flow actor adapter를 가져왔다.
- BC MLP actor adapter를 가져왔다.
- `il/` 아래 class/function docstring을 채웠다.
- simulator 없이 intervention routing smoke test를 추가했다.
- online/demo/intervention buffer를 비율대로 섞는 mixed replay sampler를 추가했다.
- `RLPDPolicy.from_checkpoint()` generic loader를 추가했다.
- `BCFlowPolicy.from_checkpoint()` generic loader를 추가했다.
- RLPD/BC Flow checkpoint save-load smoke test를 추가했다.
- DAgger baseline용 learner rollout + expert relabel helper를 추가했다.
- BC Flow에 `target_action_key` 옵션을 추가했다.

## 현재 검증 명령어

```bash
conda run -n il python -m compileall -q il scripts
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n il python scripts/smoke_test.py
```

현재 smoke 결과:

```text
gate/replay smoke ok
intervention routing smoke ok
mixed replay sampling smoke ok
dagger baseline smoke ok
rlpd smoke ok
rlpd policy checkpoint smoke ok
bc mlp smoke ok
bc flow smoke ok
bc flow policy checkpoint smoke ok
```

## 다음 구현 순서

최신 순서는 [IMPLEMENTATION_TODO.md](IMPLEMENTATION_TODO.md)를 따른다. 현재 기준으로 우선순위가 높은
작업은 다음이다.

1. `CODE_REVIEW_2026-05-26.md`의 남은 P0/P1 항목을 작은 단위로 처리한다.
2. offline demo/prefill dataset adapter와 canonicalization interface를 추가한다.
3. replay save/load round-trip test를 실제 env 산출물까지 포함해 보강한다.
4. DAgger update-on smoke와 residual large config runtime을 검증한다.
5. 새 gate family를 추가하기 전에 `ControllerGate` runtime contract와 필요 시 `GateContext`를 정리한다.

## 아직 안 된 것

- replay buffer save/load round-trip test
- dataset adapter / canonicalization interface
- `update_interval`, `updates_per_step`, `save_final`, `keep_last` 등 일부 public config field의 runtime 반영
- learner/expert 일반 action chunk queue
- image observation policy/network 학습 경로

## Git 운영

- 이 repo는 `/home/junhyeong/repos/intervention_learning`에서 독립 git으로 관리한다.
- remote는 아직 연결하지 않는다.
- GitHub SSH 설정이 끝나면 그때 remote를 추가하고 push한다.
- weight, replay, video, log, wandb 산출물은 절대 커밋하지 않는다.
- 큰 변경은 작은 커밋으로 쪼갠다.
